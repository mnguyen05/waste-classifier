#!/usr/bin/env python3
"""
Raspberry Pi Camera: capture one frame, then either:

  --server URL   Upload JPEG to your inference server (runs server.py + model there).
  (default)      Run the model on the Pi (needs checkpoints/model.pt on the Pi).

  --ir-loop      Wait on a GPIO IR sensor; each trigger captures one frame and runs
                 inference (same as one-shot). Requires gpiozero on the Pi.

Install camera stack on the Pi:

  sudo apt update && sudo apt install -y python3-picamera2

IR / GPIO (typical FC-51 style: D0 goes LOW when object detected — default polarity):

  sudo apt install -y python3-gpiozero
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

from label_mapping import CLASS_NAMES


def capture_frame(warmup: float) -> Image.Image:
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise SystemExit(
            "picamera2 is not available. On Raspberry Pi OS install with:\n"
            "  sudo apt update && sudo apt install -y python3-picamera2\n"
        ) from e

    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(warmup)
    arr = picam2.capture_array("main")
    picam2.stop()
    picam2.close()
    return Image.fromarray(arr, mode="RGB")


def upload_and_predict(server_url: str, image: Image.Image, timeout: float) -> dict:
    import requests

    url = server_url.rstrip("/") + "/predict"
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_bytes = buf.getvalue()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    print(
        f"[upload] POST {url}  ({len(image_bytes)} bytes JPEG)",
        flush=True,
    )
    try:
        # Force direct LAN connection; avoid proxy env vars (HTTP_PROXY/HTTPS_PROXY)
        # which can break local Pi -> Mac response handling.
        with requests.Session() as sess:
            sess.trust_env = False
            r = sess.post(
                url,
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                timeout=timeout,
            )
            r.raise_for_status()
    except requests.exceptions.RequestException as e:
        extra = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            extra = f" | HTTP {resp.status_code}: {resp.text[:400]!r}"
        raise RuntimeError(f"Upload failed: {e!s}{extra}") from e
    try:
        payload = r.json()
    except Exception as e:
        raise RuntimeError(
            f"Server did not return JSON (first 200 chars): {r.text[:200]!r}"
        ) from e
    server_hash = payload.get("image_sha256")
    if not server_hash:
        raise RuntimeError("Server response missing image_sha256")
    if server_hash != image_sha256:
        raise RuntimeError(
            "Server hash mismatch: captured image was not the one inferred"
        )
    payload["client_image_sha256"] = image_sha256
    payload["verified_image_match"] = True
    return payload


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def invoke_sort_hook(spec: str, payload: dict) -> None:
    """
    Call user code after a successful prediction. spec is "module:function".
    The Pi already has `payload` from the HTTP response (no second network hop).
    """
    mod_part, sep, func_part = spec.partition(":")
    mod_part, func_part = mod_part.strip(), func_part.strip()
    if not sep or not mod_part or not func_part:
        raise ValueError(
            f"Invalid --sort-hook {spec!r}; use MODULE:FUNCTION "
            "(e.g. my_seesaw:apply_sort_decision). Run from the directory that contains the module or set PYTHONPATH."
        )
    mod = importlib.import_module(mod_part)
    fn = getattr(mod, func_part)
    if not callable(fn):
        raise TypeError(f"{spec!r}: {func_part!r} is not callable")
    print(f"[sort-hook] {spec}(payload)  decision={payload.get('decision')!r}", flush=True)
    fn(payload)


def run_once(args: argparse.Namespace) -> None:
    """Single capture → save JPEG → upload or local infer (one cycle)."""
    args.captures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    image_path = args.captures_dir / f"{stamp}.jpg"
    json_path = args.results_dir / f"{stamp}.json"

    if getattr(args, "verbose", False):
        print("[IR] capturing frame...", flush=True)
    im = capture_frame(args.warmup)
    im.save(image_path, format="JPEG", quality=92)
    print(f"Saved frame to {image_path.resolve()}", flush=True)

    if args.capture_only:
        print("Capture-only mode complete.", flush=True)
        return

    if args.server:
        result = upload_and_predict(args.server, im, args.timeout)
        text = json.dumps(result, indent=2)
        print(text)
        json_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved JSON to {json_path.resolve()}", flush=True)
        if getattr(args, "sort_hook", None):
            try:
                invoke_sort_hook(args.sort_hook, result)
            except Exception as hook_ex:
                print(f"[sort-hook] error: {hook_ex!r}", flush=True)
                traceback.print_exc()
        return

    from predict import classify_pil, load_checkpoint
    from util_device import get_torch_device

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Missing checkpoint: {args.checkpoint}\n"
            "Copy model.pt to checkpoints/ or use --server to run inference elsewhere."
        )

    device = get_torch_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 224))
    label, conf, probs = classify_pil(model, im, device, img_size)

    print(f"Prediction: {label} ({conf:.1%} confidence)")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name}: {p:.1%}")

    payload = {
        "label": label,
        "confidence": conf,
        "trash_probability": probs[0],
        "recycling_probability": probs[1],
        "decision": label,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved JSON to {json_path.resolve()}", flush=True)
    if getattr(args, "sort_hook", None):
        try:
            invoke_sort_hook(args.sort_hook, payload)
        except Exception as hook_ex:
            print(f"[sort-hook] error: {hook_ex!r}", flush=True)
            traceback.print_exc()


class _IrEdge:
    """gpiozero Button: active-low (pull_up=True) or active-high (pull_up=False)."""

    def __init__(self, btn) -> None:
        self._btn = btn

    @property
    def is_active(self) -> bool:
        return bool(self._btn.is_pressed)

    def wait_for_active(self) -> None:
        self._btn.wait_for_press()

    def wait_for_inactive(self) -> None:
        self._btn.wait_for_release()

    def close(self) -> None:
        self._btn.close()


def _open_ir_sensor(gpio_pin: int, polarity: str, debounce: float) -> _IrEdge:
    """Use Button only: avoids gpiozero 2 + lgpio PinInvalidState on DigitalInputDevice+pull_up+active_state.

    low:  pull_up=True  — typical IR module sinks GPIO to GND when object detected.
    high: pull_up=False — sensor drives GPIO HIGH when active (needs a defined LOW when idle).
    """
    from gpiozero import Button

    if polarity == "low":
        dev = Button(gpio_pin, pull_up=True, bounce_time=debounce)
    else:
        dev = Button(gpio_pin, pull_up=False, bounce_time=debounce)
    return _IrEdge(dev)


def run_gpio_monitor(args: argparse.Namespace) -> None:
    """Print when GPIO sees press/release — use to verify D0→Pi wiring vs board LED."""
    from gpiozero import Button

    if args.ir_polarity == "low":
        b = Button(args.gpio_pin, pull_up=True, bounce_time=0.05)
    else:
        b = Button(args.gpio_pin, pull_up=False, bounce_time=0.05)

    print(
        f"GPIO monitor: BCM {args.gpio_pin}, polarity={args.ir_polarity!r}. "
        "Block the IR beam — is_pressed should flip. Ctrl+C to quit.",
        flush=True,
    )
    print(f"  initial is_pressed={b.is_pressed}", flush=True)
    prev = b.is_pressed
    try:
        while True:
            cur = b.is_pressed
            if cur != prev:
                print(
                    time.strftime("%H:%M:%S"),
                    f"  is_pressed={cur}  (GPIO sees {'ACTIVE' if cur else 'idle'})",
                    flush=True,
                )
                prev = cur
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nMonitor stopped.", flush=True)
    finally:
        b.close()


def run_ir_loop(args: argparse.Namespace) -> None:
    """Block on GPIO IR sensor; each activation runs run_once (capture + infer)."""
    if args.capture_only:
        raise SystemExit("--ir-loop cannot be used with --capture-only")

    if not args.server:
        if not args.checkpoint.is_file():
            raise SystemExit(
                "--ir-loop needs --server URL for Mac inference, or a local "
                f"checkpoint at {args.checkpoint}"
            )

    try:
        import gpiozero  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "gpiozero is required for --ir-loop. On Raspberry Pi OS:\n"
            "  sudo apt install -y python3-gpiozero\n"
            "  or: pip install gpiozero"
        ) from e

    sensor = _open_ir_sensor(args.gpio_pin, args.ir_polarity, args.ir_debounce)

    print(
        f"IR loop: GPIO BCM {args.gpio_pin}, active when pin is "
        f"{'HIGH' if args.ir_polarity == 'high' else 'LOW'}, "
        f"bounce={args.ir_debounce}s, settle={args.ir_settle}s. Ctrl+C to stop.",
        flush=True,
    )

    if sensor.is_active:
        print("[IR] sensor already active — waiting for beam to clear...", flush=True)
        sensor.wait_for_inactive()

    try:
        while True:
            sensor.wait_for_active()
            print("[IR] triggered", flush=True)
            if args.ir_settle > 0:
                time.sleep(args.ir_settle)
            try:
                run_once(args)
            except Exception as ex:
                print(f"[IR] cycle error: {ex!r}", flush=True)
                traceback.print_exc()
            sensor.wait_for_inactive()
            if args.ir_cooldown > 0:
                time.sleep(args.ir_cooldown)
    except KeyboardInterrupt:
        print("\nIR loop stopped.", flush=True)
    finally:
        sensor.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Pi camera → upload to server OR classify locally"
    )
    ap.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="Inference server base URL, e.g. http://192.168.1.50:8000 — uploads "
        "the frame to POST /predict (no PyTorch needed on the Pi)",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "model.pt",
        help="Used only for local inference (no --server)",
    )
    ap.add_argument(
        "--captures-dir",
        type=Path,
        default=base_dir / "captures",
        help="Directory to automatically save captured photos",
    )
    ap.add_argument("--warmup", type=float, default=0.8)
    ap.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture/save one photo and exit immediately (skip upload/inference)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout (seconds) when using --server",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=base_dir / "results",
        help="Directory to automatically save prediction JSON files",
    )
    ap.add_argument(
        "--ir-loop",
        action="store_true",
        help="Poll GPIO IR sensor in a loop: each trigger captures and infers "
        "(needs gpiozero; use with --server for Mac inference)",
    )
    ap.add_argument(
        "--gpio-pin",
        type=int,
        default=17,
        metavar="BCM",
        help="BCM GPIO number wired to the IR sensor digital output (default: 17)",
    )
    ap.add_argument(
        "--ir-polarity",
        choices=("low", "high"),
        default="low",
        help="Pin voltage when object is present: low=most FC-51 modules (default), high=inverted wiring",
    )
    ap.add_argument(
        "--ir-debounce",
        type=float,
        default=0.08,
        metavar="SEC",
        help="GPIO debounce time for the IR input (default: 0.08)",
    )
    ap.add_argument(
        "--ir-settle",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Sleep after IR fires before opening camera (default: 0.05)",
    )
    ap.add_argument(
        "--ir-cooldown",
        type=float,
        default=0.3,
        metavar="SEC",
        help="Sleep after object clears before accepting next trigger (default: 0.3)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Extra logging (e.g. each IR cycle start)",
    )
    ap.add_argument(
        "--gpio-monitor",
        action="store_true",
        help="Print GPIO press/release events (no camera). Use when IR LED works but --ir-loop is silent.",
    )
    ap.add_argument(
        "--sort-hook",
        type=str,
        default=None,
        metavar="MODULE:FUNCTION",
        help="After each successful inference, call FUNCTION(payload dict). "
        "Example: my_seesaw:apply_sort_decision — put my_seesaw.py on PYTHONPATH or run from its directory.",
    )
    args = ap.parse_args()

    if args.gpio_monitor and args.ir_loop:
        raise SystemExit("Use either --gpio-monitor or --ir-loop, not both.")

    if args.gpio_monitor:
        run_gpio_monitor(args)
        return

    if args.ir_loop:
        run_ir_loop(args)
        return

    run_once(args)


if __name__ == "__main__":
    main()

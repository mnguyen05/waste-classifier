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
import inspect
import io
import json
import time
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
    r = requests.post(
        url,
        files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
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


def run_once(args: argparse.Namespace) -> None:
    """Single capture → save JPEG → upload or local infer (one cycle)."""
    args.captures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    image_path = args.captures_dir / f"{stamp}.jpg"
    json_path = args.results_dir / f"{stamp}.json"

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


class _IrEdge:
    """Same wait API for gpiozero DigitalInputDevice (2.x) or Button (1.x active-low)."""

    def __init__(self, dev, *, is_button: bool) -> None:
        self._dev = dev
        self._is_button = is_button

    @property
    def is_active(self) -> bool:
        if self._is_button:
            return bool(self._dev.is_pressed)
        return bool(self._dev.is_active)

    def wait_for_active(self) -> None:
        if self._is_button:
            self._dev.wait_for_press()
        else:
            self._dev.wait_for_active()

    def wait_for_inactive(self) -> None:
        if self._is_button:
            self._dev.wait_for_release()
        else:
            self._dev.wait_for_inactive()

    def close(self) -> None:
        self._dev.close()


def _open_ir_sensor(gpio_pin: int, polarity: str, debounce: float) -> _IrEdge:
    """Support apt's gpiozero 1.x (no active_high) and pip's gpiozero 2.x (active_state)."""
    from gpiozero import Button, DigitalInputDevice

    active_when_high = polarity == "high"
    sig = inspect.signature(DigitalInputDevice.__init__)

    if "active_state" in sig.parameters:
        dev = DigitalInputDevice(
            gpio_pin,
            pull_up=True,
            active_state=active_when_high,
            bounce_time=debounce,
        )
        return _IrEdge(dev, is_button=False)
    if "active_high" in sig.parameters:
        dev = DigitalInputDevice(
            gpio_pin,
            pull_up=True,
            active_high=active_when_high,
            bounce_time=debounce,
        )
        return _IrEdge(dev, is_button=False)

    if active_when_high:
        raise SystemExit(
            '--ir-polarity high needs gpiozero 2.x. On the Pi run:\n'
            '  pip install "gpiozero>=2"\n'
            "Or use default --ir-polarity low for typical IR modules (output LOW when object detected)."
        )

    dev = Button(gpio_pin, pull_up=True, bounce_time=debounce)
    return _IrEdge(dev, is_button=True)


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
    args = ap.parse_args()

    if args.ir_loop:
        run_ir_loop(args)
        return

    run_once(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Raspberry Pi Camera: capture one frame, then either:

  --server URL   Upload JPEG to your inference server (runs server.py + model there).
  (default)      Run the model on the Pi (needs checkpoints/model.pt on the Pi).

Install camera stack on the Pi:

  sudo apt update && sudo apt install -y python3-picamera2
"""

from __future__ import annotations

import argparse
import hashlib
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
    args = ap.parse_args()

    # One-shot mode: capture once, infer once, save artifacts, then exit.
    args.captures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


if __name__ == "__main__":
    main()

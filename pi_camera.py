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
import io
import json
import time
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
    buf.seek(0)
    r = requests.post(
        url,
        files={"file": ("frame.jpg", buf, "image/jpeg")},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
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
        "--output",
        type=Path,
        default=None,
        help="Save captured frame to this path",
    )
    ap.add_argument("--warmup", type=float, default=0.8)
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout (seconds) when using --server",
    )
    args = ap.parse_args()

    im = capture_frame(args.warmup)

    if args.output:
        im.save(args.output)
        print(f"Saved frame to {args.output}")

    if args.server:
        result = upload_and_predict(args.server, im, args.timeout)
        print(json.dumps(result, indent=2))
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


if __name__ == "__main__":
    main()

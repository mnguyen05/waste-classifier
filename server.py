#!/usr/bin/env python3
"""
HTTP API: upload a photo → trash/recycling JSON.

Run: uvicorn server:app --host 0.0.0.0 --port 8000

Later your IR-triggered device can POST the same way, or call this from a small
on-device client; the flapper can use `decision` / `label` in the response.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from label_mapping import CLASS_NAMES
from predict import classify_pil, load_checkpoint
from util_device import get_torch_device

CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "model.pt"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Missing {CHECKPOINT}. Train first or copy model.pt there."
        )
    device = get_torch_device()
    model, ckpt = load_checkpoint(CHECKPOINT, device)
    img_size = int(ckpt.get("img_size", 224))
    _state["model"] = model
    _state["device"] = device
    _state["img_size"] = img_size
    print(f"Loaded model from {CHECKPOINT} on {device}")
    yield
    _state.clear()


app = FastAPI(title="Waste classifier", lifespan=lifespan)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    trash_probability: float
    recycling_probability: float
    decision: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Expected an image file (image/jpeg, image/png, ...)")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 15 MB)")
    try:
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not read image") from None

    model = _state["model"]
    device = _state["device"]
    img_size = _state["img_size"]
    label, conf, probs = classify_pil(model, pil, device, img_size)
    return PredictResponse(
        label=label,
        confidence=conf,
        trash_probability=probs[0],
        recycling_probability=probs[1],
        decision=label,
    )

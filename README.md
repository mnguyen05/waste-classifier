# Waste classifier: trash vs recycling

PyTorch **ResNet-18** binary classifier trained on the [Recyclable and Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)–style folder layout. Categories are mapped to **trash** or **recycling** in `label_mapping.py` (adjust for your local rules).

## Setup

```bash
pip install -r requirements.txt
```

Download ImageNet ResNet-18 weights (or let torchvision cache them on first run):

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

## Train

```bash
export WASTE_DATA_ROOT="/path/to/dataset/images"
python train.py --epochs 15
# or: python train.py --data-root "/path/to/images" --epochs 15
```

Use `--from-scratch` only if you cannot use pretrained weights.

## Predict

```bash
python predict.py path/to/image.jpg
```

Checkpoint path: `checkpoints/model.pt` (not committed; produced by training).

## System architecture (camera → sort)

**Raspberry Pi + Camera Module:** run inference directly on the Pi (see below), or run `server.py` on the Pi and POST images from another process. For a **flapper**, use the printed label or the `/predict` JSON field `decision` to drive GPIO (servo / relay).

**IR trigger:** With `gpiozero` and the Pi’s GPIO wired to your IR module’s digital output, run a continuous loop that captures on each detection and POSTs to the Mac (same as one-shot):

```bash
sudo apt install -y python3-gpiozero
python3 pi_camera.py --ir-loop --server http://192.168.1.50:8000 --gpio-pin 17
```

Default `--ir-polarity low` matches common FC-51-style sensors (D0 **LOW** when an object blocks the beam). Use `--ir-polarity high` if your module is wired the other way. Tuning: `--ir-debounce`, `--ir-settle`, `--ir-cooldown`.

Flow: **IR GPIO → frame → POST /predict (Mac) → label JSON** → your actuator can read `decision` from saved `results/*.json` or you extend the script.

## Raspberry Pi camera → middleman server (recommended for your project)

Put the **heavy model** on a PC or a powerful Pi, and keep the **camera Pi** thin:

1. On the **inference machine**, run the API (needs `checkpoints/model.pt` there):

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. On the **camera Raspberry Pi**, install Picamera2 (`apt`) and the thin client deps (no PyTorch needed):

   ```bash
   sudo apt update && sudo apt install -y python3-picamera2
   pip install -r requirements-pi-client.txt
   ```

   Copy `pi_camera.py` to the Pi (or clone the repo). Then capture and upload:

   ```bash
   python3 pi_camera.py --server http://192.168.1.50:8000
   ```

   Use your server’s LAN IP. The Pi JPEGs the frame and `POST`s it to `/predict`; the response is JSON (`label`, `decision`, probabilities) for your flapper logic.

3. Captures are saved under `captures/` and server responses under `results/` automatically.

## Raspberry Pi camera → local model (no server)

If the model runs **on the same Pi** as the camera, copy `checkpoints/model.pt` to the Pi and:

```bash
sudo apt update && sudo apt install -y python3-picamera2
pip install -r requirements.txt
python3 pi_camera.py
```

No `--server`: runs PyTorch on-device like `predict.py`.

## Upload API (testing uploads from a phone or another PC)

Start the server (needs `checkpoints/model.pt` on that machine):

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Send a photo:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/photo.jpg"
```

Response example:

```json
{
  "label": "recycling",
  "confidence": 0.94,
  "trash_probability": 0.06,
  "recycling_probability": 0.94,
  "decision": "recycling"
}
```

Use `GET /health` to verify the service is up. On a LAN, use the computer’s IP instead of `localhost` from your phone.

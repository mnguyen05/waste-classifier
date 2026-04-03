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

# CrackSeg 🛠️ — U-Net Semantic Segmentation for Structural Crack Detection

An end-to-end PyTorch pipeline that detects and segments cracks in structural /
pavement images using a U-Net convolutional neural network. Built as an
independent extension of lab work on AI-based structural health monitoring,
reimplemented in PyTorch with a public dataset for full reproducibility.

## ✨ Features
- U-Net architecture built from scratch (encoder-decoder with skip connections)
- Data augmentation (flips, rotation, brightness/contrast jitter)
- Training with Dice + BCE combined loss
- Metrics: IoU (Intersection over Union) and Dice score, tracked per epoch
- Checkpointing + training curve plots
- Inference script for single images or folders
- ONNX export for deployment
- Optional Gradio demo for live browser-based testing

## 📁 Project Structure
```
CrackSeg/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py         # U-Net architecture
│   ├── dataset.py        # Dataset loader + augmentations
│   ├── metrics.py        # IoU / Dice score functions
│   ├── train.py           # Training loop
│   ├── inference.py     # Run inference on new images
│   └── export_onnx.py  # Export trained model to ONNX
├── demo/
│   └── app.py                # Gradio web demo
└── data/
    ├── images/               # Put raw crack images here
    └── masks/                # Corresponding binary masks
```

## 📦 Dataset
This project is designed to work with any binary crack-segmentation dataset,
e.g.:
- [Crack500](https://github.com/fyangneil/pavement-crack-detection)
- [DeepCrack](https://github.com/yhlleo/DeepCrack)
- [CFD (Crack Forest Dataset)](https://github.com/cuilimeng/CrackForest-dataset)

Download one of these, and arrange it as:
```
data/images/img_001.jpg
data/masks/img_001.png   # same filename, binary mask (0/255)
```

## 🚀 Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train
python src/train.py --data_dir data --epochs 50 --batch_size 8 --lr 1e-4

# 3. Run inference on a new image
python src/inference.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg

# 4. (Optional) Export to ONNX
python src/export_onnx.py --checkpoint checkpoints/best_model.pth

# 5. (Optional) Launch demo
python demo/app.py
```

## 📊 Results
| Metric | Value (fill in after training) |
|--------|-------------------------------|
| IoU    | -- |
| Dice   | -- |
| Pixel Accuracy | -- |

## 🧠 Background
This project extends earlier lab work (semantic segmentation for crack
detection using a UNet CNN in MATLAB) into a fully open-source, reproducible
PyTorch pipeline trained on public data, with the goal of exploring
deployment-ready structural health monitoring tools.

## 📝 License
MIT

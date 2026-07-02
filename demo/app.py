"""
Gradio demo: upload an image, see the predicted crack mask + overlay.

Usage:
    python demo/app.py
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from model import UNet  # noqa: E402

CHECKPOINT_PATH = "checkpoints/best_model.pth"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()


def segment_crack(image: np.ndarray):
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    norm = (resized / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h))

    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return mask, blended


demo = gr.Interface(
    fn=segment_crack,
    inputs=gr.Image(label="Upload a structural / pavement image"),
    outputs=[
        gr.Image(label="Predicted Crack Mask"),
        gr.Image(label="Overlay"),
    ],
    title="CrackSeg — AI Crack Detection Demo",
    description="Upload an image of a wall, pavement, or structure to detect cracks using a U-Net segmentation model.",
)

if __name__ == "__main__":
    demo.launch()

"""
Run inference with a trained CrackSeg model on a single image or folder of images.

Usage:
    python src/inference.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
    python src/inference.py --checkpoint checkpoints/best_model.pth --folder path/to/images/ --out_dir results/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from model import UNet

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess(image, img_size=256):
    image = cv2.resize(image, (img_size, img_size))
    image_norm = (image / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def predict(model, image_path, device, img_size=256, threshold=0.5):
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    tensor = preprocess(image_rgb, img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h))

    overlay = image_bgr.copy()
    overlay[mask > 0] = [0, 0, 255]  # highlight crack pixels in red
    blended = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

    return mask, blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image:
        paths = [Path(args.image)]
    elif args.folder:
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts]
    else:
        raise ValueError("Provide either --image or --folder")

    for path in paths:
        mask, blended = predict(model, path, device, args.img_size, args.threshold)
        cv2.imwrite(os.path.join(args.out_dir, f"{path.stem}_mask.png"), mask)
        cv2.imwrite(os.path.join(args.out_dir, f"{path.stem}_overlay.png"), blended)
        print(f"Saved results for {path.name}")


if __name__ == "__main__":
    main()

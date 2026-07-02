"""
Dataset loader for image / binary-mask pairs used in crack segmentation.
Expects:
    data_dir/images/*.jpg (or .png)
    data_dir/masks/*.png   (same filename stem, single-channel 0/255)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms(img_size=256):
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class CrackDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=256, transform=None):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in exts]
        )

        # simple 90/10 train/val split (deterministic)
        n_val = max(1, int(0.1 * len(self.image_paths)))
        if split == "train":
            self.image_paths = self.image_paths[:-n_val] if n_val > 0 else self.image_paths
        elif split == "val":
            self.image_paths = self.image_paths[-n_val:]

        self.img_size = img_size
        self.transform = transform or (
            get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
        )

    def __len__(self):
        return len(self.image_paths)

    def _find_mask(self, img_path: Path):
        stem = img_path.stem
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = self.mask_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for {img_path.name}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self._find_mask(img_path)

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # binarize

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return image, mask

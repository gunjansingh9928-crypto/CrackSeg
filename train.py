"""
Training loop for CrackSeg U-Net.

Usage:
    python src/train.py --data_dir data --epochs 50 --batch_size 8 --lr 1e-4
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrackDataset
from metrics import DiceBCELoss, dice_score, iou_score
from model import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = CrackDataset(args.data_dir, split="train", img_size=args.img_size)
    val_ds = CrackDataset(args.data_dir, split="val", img_size=args.img_size)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_iou += iou_score(outputs, masks) * images.size(0)
                val_dice += dice_score(outputs, masks) * images.size(0)
        val_loss /= len(val_ds)
        val_iou /= len(val_ds)
        val_dice /= len(val_ds)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_IoU={val_iou:.4f} val_Dice={val_dice:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"  -> New best model saved (IoU={best_iou:.4f})")

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "last_model.pth"))

    # plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["val_iou"], label="IoU")
    axes[1].plot(history["val_dice"], label="Dice")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "training_curves.png"))
    print("Training complete. Curves saved to checkpoints/training_curves.png")


if __name__ == "__main__":
    main()

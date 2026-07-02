"""
Segmentation metrics: IoU, Dice score, pixel accuracy.
"""

import torch


def iou_score(pred_logits, target, threshold=0.5, eps=1e-7):
    """
    pred_logits: (B, 1, H, W) raw logits
    target:      (B, 1, H, W) binary {0,1}
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def dice_score(pred_logits, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return dice.mean().item()


def pixel_accuracy(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + BCE loss — works well for imbalanced crack masks
    (cracks are usually a small fraction of total pixels)."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, pred_logits, target, eps=1e-7):
        bce_loss = self.bce(pred_logits, target)

        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        )
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

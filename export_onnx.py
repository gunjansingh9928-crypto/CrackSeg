"""
Export a trained CrackSeg U-Net checkpoint to ONNX for deployment.

Usage:
    python src/export_onnx.py --checkpoint checkpoints/best_model.pth --out crackseg.onnx
"""

import argparse

import torch

from model import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints/crackseg.onnx")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )
    print(f"Model exported to {args.out}")


if __name__ == "__main__":
    main()

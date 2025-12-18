"""
Evaluate a checkpoint on the test split and report segmentation metrics.

Usage (from repo root):
    py -3 run_test.py --checkpoint lightning_logs_archive\\version_2\\checkpoints\\last.ckpt

Options:
    --root <path>          Root dataset dir (default: Config.root_dir).
    --batch-size <int>     Batch size for test loader (default: Config.val_batch_size).
    --threshold <float>    Sigmoid threshold for metrics (default: 0.5).
    --save-csv <path>      Optional path to save a one-line CSV with metrics.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import yaml

from lightning_train import Config, LitUNet, KvasirDataset, build_transforms, set_seed
from loss import calculate_metrics

# Edit this to point to the checkpoint you want by default.
# Use forward slashes or a raw string to avoid Windows escape issues.
DEFAULT_CHECKPOINT = Path("lightning_logs_archive/version_0\checkpoints\last.ckpt")


def main():
    parser = argparse.ArgumentParser(description="Run test evaluation for a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to .ckpt file (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--root", type=Path, default=Path(Config().root_dir), help="Dataset root dir.")
    parser.add_argument("--batch-size", type=int, default=Config().val_batch_size, help="Test batch size.")
    parser.add_argument("--num-workers", type=int, default=Config().num_workers, help="DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions.")
    parser.add_argument("--save-csv", type=Path, help="Optional CSV output for the aggregated metrics.")
    args = parser.parse_args()

    # Load config from hparams.yaml next to the checkpoint (version_*/hparams.yaml) if present.
    cfg = Config(root_dir=str(args.root))
    hparams_path = args.checkpoint.parent.parent / "hparams.yaml"
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.checkpoint.suffix != ".ckpt":
        raise SystemExit(f"Expected a .ckpt file, got: {args.checkpoint}")
    if hparams_path.exists():
        with hparams_path.open("r", encoding="utf-8") as f:
            hp = yaml.safe_load(f) or {}
        if isinstance(hp, dict) and "cfg" in hp and isinstance(hp["cfg"], dict):
            for k, v in hp["cfg"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    # Always respect the CLI root override.
    cfg.root_dir = str(args.root)

    set_seed(cfg.seed)

    _, eval_tfm = build_transforms(cfg.img_size)
    test_ds = KvasirDataset(cfg.root_dir, phase="test", transform=eval_tfm)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitUNet.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.to(device)
    model.eval()

    sums: Dict[str, float] = dict(dice=0.0, iou=0.0, precision=0.0, recall=0.0)
    total = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            main_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            d, iou, p, r = calculate_metrics(main_out, masks, threshold=args.threshold)
            bs = images.size(0)
            total += bs
            sums["dice"] += d * bs
            sums["iou"] += iou * bs
            sums["precision"] += p * bs
            sums["recall"] += r * bs

    if total == 0:
        raise SystemExit("Test loader is empty. Check dataset paths.")

    metrics = {k: v / total for k, v in sums.items()}
    print(f"Evaluated {total} samples on {args.checkpoint}")
    print(
        "dice: {dice:.4f}, iou: {iou:.4f}, precision: {precision:.4f}, recall: {recall:.4f}".format(
            **metrics
        )
    )

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "dice", "iou", "precision", "recall"])
            writer.writeheader()
            writer.writerow({"checkpoint": str(args.checkpoint), **metrics})
        print(f"Wrote metrics to {args.save_csv}")


if __name__ == "__main__":
    main()

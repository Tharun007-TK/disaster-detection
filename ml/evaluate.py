"""Evaluate SiameseDamageNet on BRIGHT val split.

Loads best.pth, computes per-class IoU/precision/recall/F1, confusion matrix,
saves JSON report + confusion-matrix PNG + sample prediction grids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import (
    NUM_CLASSES,
    BrightDataset,
    build_val_transform,
    discover_pairs,
    split_pairs,
)
from ml.model import SiameseDamageNet


CLASS_NAMES = ["No Damage", "Minor", "Major", "Destroyed"]
CLASS_COLORS = np.array(
    [
        [0, 128, 0],      # green
        [255, 255, 0],    # yellow
        [255, 140, 0],    # orange
        [220, 20, 60],    # crimson
    ],
    dtype=np.uint8,
)


@torch.no_grad()
def build_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for batch in tqdm(loader, desc="eval", leave=False):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(pre, post)
        pred = logits.argmax(dim=1)
        k = (mask >= 0) & (mask < num_classes)
        inds = num_classes * mask[k].to(torch.int64) + pred[k]
        mat += torch.bincount(inds, minlength=num_classes ** 2).reshape(
            num_classes, num_classes
        )
    return mat


def _nan_to_none(x: float) -> float | None:
    return None if np.isnan(x) else round(float(x), 6)


def metrics_from_cm(cm: np.ndarray) -> dict:
    cm_f = cm.astype(np.float64)
    tp = np.diag(cm_f)
    fp = cm_f.sum(axis=0) - tp
    fn = cm_f.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            np.nan,
        )

    pixel_acc = float(tp.sum() / max(1.0, cm_f.sum()))
    miou = float(np.nanmean(iou))

    per_class = []
    for c in range(cm.shape[0]):
        per_class.append(
            {
                "class": c,
                "name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                "iou": _nan_to_none(iou[c]),
                "precision": _nan_to_none(precision[c]),
                "recall": _nan_to_none(recall[c]),
                "f1": _nan_to_none(f1[c]),
                "support_px": int(cm_f.sum(axis=1)[c]),
            }
        )

    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "per_class": per_class,
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion Matrix (pixel counts)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, f"{cm[i,j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = CLASS_COLORS[c]
    return out


@torch.no_grad()
def save_sample_grid(
    model: torch.nn.Module,
    dataset: BrightDataset,
    device: torch.device,
    n_samples: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n = min(n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    for idx in indices:
        s = dataset[int(idx)]
        pre = s["pre"].unsqueeze(0).to(device)
        post = s["post"].unsqueeze(0).to(device)
        mask = s["mask"].numpy()
        logits = model(pre, post)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        pre_img = s["pre"].permute(1, 2, 0).cpu().numpy()
        post_img = s["post"][0].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(np.clip(pre_img, 0, 1)); axes[0].set_title("Pre (RGB)")
        axes[1].imshow(post_img, cmap="gray");  axes[1].set_title("Post (SAR)")
        axes[2].imshow(colorize_mask(mask));    axes[2].set_title("Ground truth")
        axes[3].imshow(colorize_mask(pred));    axes[3].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(s["stem"], fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{s['stem']}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--ckpt", default="./ml/checkpoints/best.pth")
    parser.add_argument("--out-dir", default="./ml/checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading weights: {args.ckpt}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(data_dir)
    _, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    val_ds = BrightDataset(data_dir, pairs=val_pairs, transform=build_val_transform(args.img_size))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Val samples: {len(val_ds)}")

    model = SiameseDamageNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    cm_tensor = build_confusion_matrix(model, val_loader, device, NUM_CLASSES)
    cm = cm_tensor.cpu().numpy()
    metrics = metrics_from_cm(cm)
    metrics["num_val_samples"] = len(val_ds)
    metrics["checkpoint"] = str(args.ckpt)

    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {report_path}")

    cm_png = out_dir / "confusion_matrix.png"
    save_confusion_matrix_png(cm, cm_png)
    print(f"wrote {cm_png}")

    if not args.no_samples:
        samples_dir = out_dir / "samples"
        save_sample_grid(model, val_ds, device, args.n_samples, samples_dir)
        print(f"wrote {args.n_samples} samples -> {samples_dir}")

    print("\n=== Summary ===")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  mIoU:        {metrics['miou']:.4f}")
    print(f"  Pixel acc:   {metrics['pixel_acc']:.4f}")
    print("  Per-class:")
    for row in metrics["per_class"]:
        iou = f"{row['iou']:.4f}" if row["iou"] is not None else "  n/a "
        f1 = f"{row['f1']:.4f}" if row["f1"] is not None else "  n/a "
        prec = f"{row['precision']:.4f}" if row["precision"] is not None else "  n/a "
        rec = f"{row['recall']:.4f}" if row["recall"] is not None else "  n/a "
        print(f"    {row['class']} {row['name']:<11} IoU {iou}  F1 {f1}  P {prec}  R {rec}")


if __name__ == "__main__":
    main()

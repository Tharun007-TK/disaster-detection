"""Train SiameseDamageNet on BRIGHT.

AdamW lr=1e-4, CrossEntropyLoss, mixed precision (CUDA only).
Best checkpoint selected by val mIoU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import (
    NUM_CLASSES,
    BrightDataset,
    ScenePair,
    build_train_transform,
    build_val_transform,
    discover_pairs,
    split_pairs,
)
from ml.model import SiameseDamageNet


def compute_class_pixel_counts(
    pairs: list[ScenePair], num_classes: int
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in tqdm(pairs, desc="class-freq", leave=False):
        with rasterio.open(p.mask_path) as src:
            arr = src.read(1)
        u, c = np.unique(arr, return_counts=True)
        for cls, n in zip(u.tolist(), c.tolist()):
            if 0 <= cls < num_classes:
                counts[cls] += n
    return counts


def class_weights_median_freq(counts: np.ndarray) -> np.ndarray:
    freqs = counts.astype(np.float64) / max(1, counts.sum())
    nonzero = freqs[freqs > 0]
    med = float(np.median(nonzero)) if nonzero.size else 1.0
    weights = np.where(freqs > 0, med / (freqs + 1e-12), 1.0)
    return weights


def resolve_class_weights(
    arg: str,
    pairs: list[ScenePair],
    num_classes: int,
    cache_path: Path,
) -> torch.Tensor | None:
    if arg == "off":
        return None
    if arg == "auto":
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            counts = np.array(cached["pixel_counts"], dtype=np.int64)
            weights = np.array(cached["weights"], dtype=np.float64)
            print(f"loaded cached class weights: {weights.tolist()}")
        else:
            counts = compute_class_pixel_counts(pairs, num_classes)
            weights = class_weights_median_freq(counts)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "pixel_counts": counts.tolist(),
                "weights": weights.tolist(),
                "num_samples": len(pairs),
            }, indent=2))
            print(f"computed class weights: {weights.tolist()}")
            print(f"  pixel counts: {counts.tolist()}")
            print(f"  cached to {cache_path}")
        return torch.tensor(weights, dtype=torch.float32)
    try:
        vals = [float(x) for x in arg.split(",")]
    except ValueError as e:
        raise ValueError(f"--class-weights must be off|auto|csv, got {arg!r}") from e
    if len(vals) != num_classes:
        raise ValueError(
            f"expected {num_classes} weights in CSV, got {len(vals)}: {vals}"
        )
    return torch.tensor(vals, dtype=torch.float32)


class ConfusionMatrix:
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        k = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[k].to(torch.int64) + pred[k]
        self.mat += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, float | list[float]]:
        mat = self.mat.float()
        tp = torch.diag(mat)
        fp = mat.sum(dim=0) - tp
        fn = mat.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
        pixel_acc = tp.sum() / mat.sum().clamp(min=1)
        return {
            "iou_per_class": iou.cpu().tolist(),
            "miou": float(torch.nanmean(iou).item()),
            "pixel_acc": float(pixel_acc.item()),
        }


def build_loaders(
    data_dir: Path,
    batch_size: int,
    img_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    pairs = discover_pairs(data_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=val_ratio, seed=seed)
    train_ds = BrightDataset(data_dir, pairs=train_pairs, transform=build_train_transform(img_size))
    val_ds = BrightDataset(data_dir, pairs=val_pairs, transform=build_val_transform(img_size))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, use_amp) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(pre, post)
            loss = loss_fn(logits, mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = pre.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp) -> dict[str, float | list[float]]:
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, device=device)
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(pre, post)
            loss = loss_fn(logits, mask)
        pred = logits.argmax(dim=1)
        cm.update(pred, mask)
        bs = pre.size(0)
        total += loss.item() * bs
        n += bs
    metrics = cm.compute()
    metrics["loss"] = total / max(1, n)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "./ml/checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--class-weights", default="off",
        help="off | auto | CSV like '1.0,2.0,5.0,3.0'",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"device: {device}  amp: {use_amp}")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(
        data_dir, args.batch_size, args.img_size, args.val_ratio,
        args.num_workers, args.seed,
    )

    model = SiameseDamageNet(
        num_classes=NUM_CLASSES, pretrained=not args.no_pretrained
    ).to(device)

    pairs = discover_pairs(data_dir)
    train_pairs, _ = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    weights_tensor = resolve_class_weights(
        args.class_weights, train_pairs, NUM_CLASSES,
        cache_path=ckpt_dir / "class_weights.json",
    )
    if weights_tensor is not None:
        weights_tensor = weights_tensor.to(device)
        print(f"using weighted CE: weights={weights_tensor.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(device="cuda", enabled=use_amp)

    best_miou = -1.0
    history: list[dict] = []
    metrics_path = ckpt_dir / "metrics.json"
    best_path = ckpt_dir / "best.pth"
    last_path = ckpt_dir / "last.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, use_amp)
        val = validate(model, val_loader, loss_fn, device, use_amp)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(float(val["loss"]), 6),
            "val_miou": round(float(val["miou"]), 6),
            "val_pixel_acc": round(float(val["pixel_acc"]), 6),
            "val_iou_per_class": [round(x, 6) if x == x else None for x in val["iou_per_class"]],
            "time_sec": round(dt, 1),
        }
        history.append(row)
        print(
            f"ep {epoch:03d}  train {train_loss:.4f}  "
            f"val {float(val['loss']):.4f}  "
            f"mIoU {float(val['miou']):.4f}  "
            f"pxAcc {float(val['pixel_acc']):.4f}  "
            f"({dt:.1f}s)"
        )
        metrics_path.write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), last_path)

        if val["miou"] > best_miou:
            best_miou = float(val["miou"])
            torch.save(model.state_dict(), best_path)
            print(f"  best mIoU -> {best_miou:.4f}  saved {best_path}")

    print(f"done. best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()

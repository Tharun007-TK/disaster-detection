"""BRIGHT GeoTIFF dataset for Siamese damage segmentation.

Pre-event: 3-band RGB uint8.
Post-event: 1-band SAR uint8.
Target: 1-band uint8, values 0-3 (No/Minor/Major/Destroyed).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


PRE_SUFFIX = "_pre_disaster.tif"
POST_SUFFIX = "_post_disaster.tif"
TARGET_SUFFIX = "_building_damage.tif"
NUM_CLASSES = 4


@dataclass(frozen=True)
class ScenePair:
    stem: str
    pre_path: Path
    post_path: Path
    mask_path: Path


def discover_pairs(data_dir: Path) -> list[ScenePair]:
    pre_dir = data_dir / "pre-event"
    post_dir = data_dir / "post-event"
    tgt_dir = data_dir / "target"

    pre_stems = {
        f.name[: -len(PRE_SUFFIX)]
        for f in pre_dir.iterdir()
        if f.name.endswith(PRE_SUFFIX)
    }
    post_stems = {
        f.name[: -len(POST_SUFFIX)]
        for f in post_dir.iterdir()
        if f.name.endswith(POST_SUFFIX)
    }
    tgt_stems = {
        f.name[: -len(TARGET_SUFFIX)]
        for f in tgt_dir.iterdir()
        if f.name.endswith(TARGET_SUFFIX)
    }

    common = sorted(pre_stems & post_stems & tgt_stems)
    return [
        ScenePair(
            stem=s,
            pre_path=pre_dir / f"{s}{PRE_SUFFIX}",
            post_path=post_dir / f"{s}{POST_SUFFIX}",
            mask_path=tgt_dir / f"{s}{TARGET_SUFFIX}",
        )
        for s in common
    ]


def _read_geotiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()  # [C, H, W]
    return arr


def build_train_transform(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"post": "image", "mask": "mask"},
    )


def build_val_transform(img_size: int = 1024) -> A.Compose:
    return A.Compose(
        [A.CenterCrop(height=img_size, width=img_size)],
        additional_targets={"post": "image", "mask": "mask"},
    )


class BrightDataset(Dataset):
    """Matched pre/post/mask triples from the BRIGHT dataset.

    Pre normalized to float32 [0,1] as 3-ch RGB.
    Post normalized to float32 [0,1] as 1-ch SAR.
    Mask returned as int64 class indices.
    """

    def __init__(
        self,
        data_dir: os.PathLike[str] | str,
        pairs: Optional[list[ScenePair]] = None,
        transform: Optional[A.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.pairs = pairs if pairs is not None else discover_pairs(self.data_dir)
        self.transform = transform
        if not self.pairs:
            raise RuntimeError(f"No matched pre/post/mask triples under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        pre = _read_geotiff(pair.pre_path)          # [3, H, W]
        post = _read_geotiff(pair.post_path)        # [1, H, W]
        mask = _read_geotiff(pair.mask_path)[0]     # [H, W]

        pre_hwc = np.transpose(pre, (1, 2, 0))      # [H, W, 3]
        post_hwc = np.transpose(post, (1, 2, 0))    # [H, W, 1]

        if self.transform is not None:
            out = self.transform(image=pre_hwc, post=post_hwc, mask=mask)
            pre_hwc = out["image"]
            post_hwc = out["post"]
            mask = out["mask"]

        pre_t = torch.from_numpy(np.ascontiguousarray(np.transpose(pre_hwc, (2, 0, 1)))).float() / 255.0
        post_t = torch.from_numpy(np.ascontiguousarray(np.transpose(post_hwc, (2, 0, 1)))).float() / 255.0
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return {"pre": pre_t, "post": post_t, "mask": mask_t, "stem": pair.stem}


def split_pairs(
    pairs: list[ScenePair],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[ScenePair], list[ScenePair]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    args = parser.parse_args()

    pairs = discover_pairs(Path(args.data_dir))
    print(f"Matched triples: {len(pairs)}")
    train_pairs, val_pairs = split_pairs(pairs)
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    ds = BrightDataset(args.data_dir, pairs=train_pairs[:4], transform=build_train_transform(512))
    sample = ds[0]
    print(f"Sample stem: {sample['stem']}")
    print(f"  pre:  {tuple(sample['pre'].shape)}  dtype={sample['pre'].dtype}  "
          f"range=[{sample['pre'].min():.3f}, {sample['pre'].max():.3f}]")
    print(f"  post: {tuple(sample['post'].shape)} dtype={sample['post'].dtype}  "
          f"range=[{sample['post'].min():.3f}, {sample['post'].max():.3f}]")
    print(f"  mask: {tuple(sample['mask'].shape)} dtype={sample['mask'].dtype}  "
          f"unique={torch.unique(sample['mask']).tolist()}")

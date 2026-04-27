"""Single pre/post pair inference for SiameseDamageNet.

Use as CLI or import `run_inference` / `load_model` from backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import NUM_CLASSES
from ml.model import SiameseDamageNet


CLASS_NAMES = ["No Damage", "Minor", "Major", "Destroyed"]
CLASS_COLORS = np.array(
    [
        [0, 128, 0],
        [255, 255, 0],
        [255, 140, 0],
        [220, 20, 60],
    ],
    dtype=np.uint8,
)


def load_model(
    ckpt_path: os.PathLike[str] | str,
    device: Optional[torch.device] = None,
    num_classes: int = NUM_CLASSES,
) -> tuple[SiameseDamageNet, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseDamageNet(num_classes=num_classes, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def _read_image(path: Path) -> tuple[np.ndarray, dict]:
    """Load any image; rasterio first (preserves CRS/bounds), PIL fallback for regular photos."""
    try:
        with rasterio.open(path) as src:
            arr = src.read()  # [C, H, W]
            crs = str(src.crs) if src.crs is not None else None
            transform = src.transform
            shape = (src.height, src.width)
            b = src.bounds
            bounds = {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top}
        return arr, {"crs": crs, "transform": transform, "shape": shape, "bounds": bounds}
    except Exception:
        from PIL import Image
        img = Image.open(path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[np.newaxis]          # grayscale → [1, H, W]
        else:
            arr = arr.transpose(2, 0, 1)  # HWC → CHW [C, H, W]
        H, W = arr.shape[1], arr.shape[2]
        return arr, {"crs": None, "transform": None, "shape": (H, W), "bounds": None}


def _load_pair(pre_path: Path, post_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict, np.ndarray]:
    pre, meta = _read_image(pre_path)
    post, _ = _read_image(post_path)
    # Hold a reference to the original post array (uint8 from rasterio/PIL)
    # for the overlay render. We do NOT make an explicit `.copy()` — the
    # downstream normalisation creates a new array via `.astype`, so the
    # original buffer stays alive only via this reference.
    post_orig = post

    # Normalise pre to 3 bands (still uint8 — float cast happens on tensor entry)
    if pre.shape[0] == 4:
        pre = pre[:3]
    elif pre.shape[0] == 1:
        pre = np.repeat(pre, 3, axis=0)
    elif pre.shape[0] != 3:
        raise ValueError(f"Pre image must have 1, 3, or 4 bands, got {pre.shape[0]}")
    # Normalise post to 1 band — only here do we promote to float32 because
    # `.mean()` requires it. Reassigning `post` releases the multiband buffer
    # as soon as the new single-band float copy lands; `post_orig` keeps the
    # uint8 multiband for overlay rendering.
    if post.shape[0] != 1:
        post = post.astype(np.float32).mean(axis=0, keepdims=True)

    if pre.shape[1:] != post.shape[1:]:
        raise ValueError(f"Pre/post spatial mismatch: pre={pre.shape[1:]} post={post.shape[1:]}")

    # Float cast deferred to tensor construction. `from_numpy` shares memory
    # with the numpy buffer, but `.astype(float32) / 255.0` allocates fresh.
    pre_t = torch.from_numpy((pre.astype(np.float32) / 255.0)).unsqueeze(0)
    if post.dtype == np.float32:
        post_t = torch.from_numpy(post / 255.0).unsqueeze(0)
    else:
        post_t = torch.from_numpy(post.astype(np.float32) / 255.0).unsqueeze(0)
    return pre_t, post_t, meta, post_orig


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = CLASS_COLORS[c]
    return out


def summarize(mask: np.ndarray) -> dict:
    total = int(mask.size)
    counts: dict[str, int] = {}
    pct: dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        n = int((mask == c).sum())
        counts[name] = n
        pct[name] = round(100.0 * n / max(1, total), 4)
    return {"pixel_counts": counts, "damage_pct": pct}


def predict_mask(
    model: SiameseDamageNet,
    pre: torch.Tensor,
    post: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run a forward pass and return an HxW uint8 class-id mask.

    Uses `torch.inference_mode()` (cheaper than `no_grad`: skips view tracking
    and version-counter bumps) and explicitly frees intermediate tensors
    before returning so peak RSS drops back to model+input footprint.
    """
    with torch.inference_mode():
        pre = pre.to(device, dtype=torch.float32)
        post = post.to(device, dtype=torch.float32)
        _, _, H, W = pre.shape

        pre_p, _ = _pad_to_multiple(pre, 32)
        post_p, _ = _pad_to_multiple(post, 32)
        # Free unpadded copies if padding made a new allocation.
        if pre_p.data_ptr() != pre.data_ptr():
            del pre
        if post_p.data_ptr() != post.data_ptr():
            del post

        logits = model(pre_p, post_p)
        del pre_p, post_p

        # Crop pad, then collapse to argmax in uint8 immediately so the
        # large float32 logits buffer can be released.
        pred_t = logits[..., :H, :W].argmax(dim=1)[0].to(torch.uint8)
        del logits

        pred = pred_t.cpu().numpy()
        del pred_t
    return pred


def save_mask_geotiff(mask: np.ndarray, out_path: Path, meta: dict) -> None:
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
    }
    if meta.get("crs"):
        profile["crs"] = meta["crs"]
    if meta.get("transform") is not None:
        profile["transform"] = meta["transform"]
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    from PIL import Image
    Image.fromarray(colorize_mask(mask)).save(out_path)


def save_overlay(orig_arr: np.ndarray, mask: np.ndarray, out_path: Path, alpha: float = 0.45) -> None:
    """Blend original image (CHW uint8) with damage color mask at given alpha."""
    from PIL import Image
    img_hwc = orig_arr.transpose(1, 2, 0) if orig_arr.ndim == 3 else orig_arr
    if img_hwc.shape[2] == 1:
        img_hwc = np.repeat(img_hwc, 3, axis=2)
    elif img_hwc.shape[2] == 4:
        img_hwc = img_hwc[:, :, :3]
    img_hwc = img_hwc.astype(np.float32)
    colored = colorize_mask(mask).astype(np.float32)
    blended = (img_hwc * (1.0 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def run_inference(
    pre_path: os.PathLike[str] | str,
    post_path: os.PathLike[str] | str,
    ckpt_path: os.PathLike[str] | str,
    out_dir: Optional[os.PathLike[str] | str] = None,
    model: Optional[SiameseDamageNet] = None,
    device: Optional[torch.device] = None,
) -> dict:
    pre_path = Path(pre_path)
    post_path = Path(post_path)
    ckpt_path = Path(ckpt_path)

    if model is None:
        model, device = load_model(ckpt_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    pre_t, post_t, meta, post_orig = _load_pair(pre_path, post_path)
    mask = predict_mask(model, pre_t, post_t, device)

    summary = summarize(mask)
    result = {
        "pre": str(pre_path),
        "post": str(post_path),
        "checkpoint": str(ckpt_path),
        "num_classes": NUM_CLASSES,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "crs": meta.get("crs"),
        "bounds": meta.get("bounds"),
        **summary,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pre_path.stem.replace("_pre_disaster", "")
        mask_png = out_dir / f"{stem}_mask.png"
        mask_tif = out_dir / f"{stem}_mask.tif"
        overlay_png = out_dir / f"{stem}_overlay.png"
        summary_json = out_dir / f"{stem}_summary.json"
        save_mask_png(mask, mask_png)
        save_mask_geotiff(mask, mask_tif, meta)
        save_overlay(post_orig, mask, overlay_png)
        summary_json.write_text(json.dumps(result, indent=2))
        result["outputs"] = {
            "mask_png": str(mask_png),
            "mask_tif": str(mask_tif),
            "overlay_png": str(overlay_png),
            "summary_json": str(summary_json),
        }
    return result


def _load_single(img_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict, np.ndarray]:
    arr, meta = _read_image(img_path)
    # Hold reference to the uint8 original for overlay rendering — no `.copy()`.
    # The `.astype(float32)` below allocates a new buffer; the original stays
    # alive via `orig_arr`.
    orig_arr = arr
    H, W = arr.shape[1], arr.shape[2]
    # Single combined float promotion + scale + mean reduction in one chain
    # so no intermediate full-size float32 multiband buffer is retained.
    post_arr = (arr.astype(np.float32) / 255.0).mean(axis=0, keepdims=True)
    pre_arr = np.zeros((3, H, W), dtype=np.float32)
    pre_t = torch.from_numpy(pre_arr).unsqueeze(0)
    post_t = torch.from_numpy(post_arr).unsqueeze(0)
    return pre_t, post_t, meta, orig_arr


def run_single_inference(
    img_path: os.PathLike[str] | str,
    ckpt_path: os.PathLike[str] | str,
    out_dir: Optional[os.PathLike[str] | str] = None,
    model: Optional[SiameseDamageNet] = None,
    device: Optional[torch.device] = None,
) -> dict:
    img_path = Path(img_path)
    ckpt_path = Path(ckpt_path)

    if model is None:
        model, device = load_model(ckpt_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    pre_t, post_t, meta, orig_arr = _load_single(img_path)
    mask = predict_mask(model, pre_t, post_t, device)

    summary = summarize(mask)
    result: dict = {
        "image": str(img_path),
        "mode": "single",
        "checkpoint": str(ckpt_path),
        "num_classes": NUM_CLASSES,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "crs": meta.get("crs"),
        "bounds": meta.get("bounds"),
        **summary,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem.replace("_post_disaster", "").replace("_pre_disaster", "")
        mask_png = out_dir / f"{stem}_single_mask.png"
        mask_tif = out_dir / f"{stem}_single_mask.tif"
        overlay_png = out_dir / f"{stem}_single_overlay.png"
        summary_json = out_dir / f"{stem}_single_summary.json"
        save_mask_png(mask, mask_png)
        save_mask_geotiff(mask, mask_tif, meta)
        save_overlay(orig_arr, mask, overlay_png)
        summary_json.write_text(json.dumps(result, indent=2))
        result["outputs"] = {
            "mask_png": str(mask_png),
            "mask_tif": str(mask_tif),
            "overlay_png": str(overlay_png),
            "summary_json": str(summary_json),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--ckpt", default=".D:/Projects/Disaster Damage Prediction/Code/v2/ml/checkpoints/best.pth")
    parser.add_argument("--out-dir", default=os.environ.get("OUTPUTS_DIR", "./outputs"))
    args = parser.parse_args()

    result = run_inference(args.pre, args.post, args.ckpt, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

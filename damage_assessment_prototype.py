"""
Multimodal Post-Disaster Building Damage Assessment
Batch Inference Pipeline — BRIGHT Dataset
RTX 4080 / Local GPU
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not installed. Run: pip install rasterio")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE         = "cuda"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
PRE_DIR        = Path("data/pre-event")
POST_DIR       = Path("data/post-event")
TARGET_DIR     = Path("data/target")
OUTPUT_DIR     = Path("outputs")
WEIGHTS_PATH   = Path("weights/damage_classifier.pt")

MIN_AREA   = 300
MAX_AREA   = 20000
MAX_ASPECT = 5.0

CLASS_NAMES  = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_COLORS = [(80,200,80),(230,180,30),(210,100,30),(200,40,40)]
GT_COLORS    = [(60,180,60),(200,160,20),(180,80,20),(180,30,30)]

OUTPUT_DIR.mkdir(exist_ok=True)

# ─── TIF LOADER ───────────────────────────────────────────────────────────────
def load_tif(path: Path, is_target: bool = False) -> np.ndarray:
    """
    Load a .tif file and return as uint8 RGB numpy array (H, W, 3).
    Handles:
      - 3-band RGB optical (pre-event)
      - 1-band or 2-band SAR (post-event) — converted to 3-ch grayscale
      - 1-band label mask (target) — returned as single channel uint8
    """
    if not HAS_RASTERIO:
        # Fallback: try cv2 (works for simple TIFs)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if is_target or img.ndim == 2:
            if img.ndim == 3:
                img = img[:, :, 0]
            return img.astype(np.uint8)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    with rasterio.open(path) as src:
        n_bands = src.count
        dtype   = src.dtypes[0]

        if is_target:
            return src.read(1).astype(np.uint8)

        if n_bands >= 3:
            # RGB optical — read bands 1,2,3
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
            img = np.stack([r, g, b], axis=-1)

        elif n_bands == 2:
            # Dual-pol SAR (VV, VH) — use VV as grayscale
            vv = src.read(1).astype(np.float32)
            img = np.stack([vv, vv, vv], axis=-1)

        else:
            # Single band — either SAR or label mask
            band = src.read(1)
            img = np.stack([band, band, band], axis=-1).astype(np.float32)

    # Normalize to 0-255 uint8
    if img.dtype != np.uint8:
        # Percentile stretch — handles SAR backscatter and optical DN values
        p2, p98 = np.percentile(img[img > 0], (2, 98)) if (img > 0).any() else (0, 1)
        p2  = max(p2, 0)
        p98 = max(p98, p2 + 1e-6)
        img = np.clip((img - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    return img

# ─── MODEL ────────────────────────────────────────────────────────────────────
class DamageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
        )
    def forward(self, pre, post):
        fp = self.encoder(pre).squeeze(-1).squeeze(-1)
        fq = self.encoder(post).squeeze(-1).squeeze(-1)
        return self.classifier(torch.cat([fp, fq], dim=1))

transform = T.Compose([
    T.Resize((64, 64)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model():
    model = DamageClassifier().to(DEVICE)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded trained weights: {WEIGHTS_PATH}")
    else:
        print("No trained weights — using ImageNet backbone (untrained head).")
    model.eval()
    return model

# ─── SAM ──────────────────────────────────────────────────────────────────────
def load_sam():
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.93,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=MIN_AREA,
    )

def is_building(mask):
    area = mask["area"]
    if area < MIN_AREA or area > MAX_AREA: return False
    x, y, w, h = mask["bbox"]
    if w < 8 or h < 8: return False
    return (max(w,h) / max(min(w,h), 1)) < MAX_ASPECT

# ─── INFERENCE ────────────────────────────────────────────────────────────────
def classify_buildings(model, mask_gen, pre_rgb, post_rgb):
    masks = mask_gen.generate(post_rgb)
    building_masks = [m for m in masks if is_building(m)]
    results = []
    with torch.no_grad():
        for md in tqdm(building_masks, desc="  Classifying", leave=False):
            x,y,w,h = [int(v) for v in md["bbox"]]
            pad=8
            x1=max(0,x-pad); y1=max(0,y-pad)
            x2=min(pre_rgb.shape[1],x+w+pad)
            y2=min(pre_rgb.shape[0],y+h+pad)
            if x2-x1<4 or y2-y1<4: continue
            pp = transform(Image.fromarray(pre_rgb[y1:y2,x1:x2])).unsqueeze(0).to(DEVICE)
            qp = transform(Image.fromarray(post_rgb[y1:y2,x1:x2])).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(pp,qp),dim=1).cpu().numpy()[0]
            cls   = int(np.argmax(probs))
            results.append({
                "mask":md,"class_idx":cls,"class_name":CLASS_NAMES[cls],
                "confidence":float(probs[cls]),"scores":probs.tolist(),
                "bbox":(x1,y1,x2,y2)
            })
    return results

# ─── OVERLAYS ─────────────────────────────────────────────────────────────────
def render_gt_overlay(image_rgb, gt_mask):
    overlay = image_rgb.astype(np.float32).copy()
    if gt_mask is None:
        return overlay.astype(np.uint8)
    
    # Ensure gt_mask is 2D
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
        
    gt = cv2.resize(gt_mask, (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    for i, color in enumerate(GT_COLORS):
        m = (gt == i)
        if m.any():
            overlay[m] = overlay[m]*0.4 + np.array(color,dtype=np.float32)*0.6
    return np.clip(overlay,0,255).astype(np.uint8)

def render_pred_overlay(image_rgb, results):
    overlay = image_rgb.astype(np.float32).copy()
    for r in results:
        seg   = r["mask"]["segmentation"]
        color = np.array(CLASS_COLORS[r["class_idx"]],dtype=np.float32)
        overlay[seg] = overlay[seg]*0.45 + color*0.55
    return np.clip(overlay,0,255).astype(np.uint8)

def make_legend(dist=None, use_gt=False):
    colors = GT_COLORS if use_gt else CLASS_COLORS
    return [
        mpatches.Patch(
            color=np.array(colors[i])/255,
            label=f"{CLASS_NAMES[i]:15s}" + (f" ({dist.get(CLASS_NAMES[i],0)})" if dist else "")
        ) for i in range(4)
    ]

def save_comparison(pre_rgb, post_rgb, gt_overlay, pred_overlay, results, stem):
    dist = Counter(r["class_name"] for r in results)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(stem, color="white", fontsize=12, fontfamily="monospace", y=0.99)

    for ax, img, title in zip(axes.flat,
        [pre_rgb, post_rgb, gt_overlay, pred_overlay],
        ["Pre-Disaster (Optical)", "Post-Disaster (SAR)",
         "Ground Truth", f"Model Prediction — {len(results)} buildings"]):
        ax.imshow(img); ax.axis("off"); ax.set_facecolor("#0d1117")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)

    axes[1][0].legend(handles=make_legend(use_gt=True), loc="lower right",
        fontsize=9, framealpha=0.85, facecolor="#1a1a2e",
        edgecolor="#3f3f46", labelcolor="white")
    axes[1][1].legend(handles=make_legend(dist=dist), loc="lower right",
        fontsize=9, framealpha=0.85, facecolor="#1a1a2e",
        edgecolor="#3f3f46", labelcolor="white")

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / f"{stem}_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out.name}")

# ─── PAIR FINDER ──────────────────────────────────────────────────────────────
def find_pairs():
    pairs = []
    for pre_path in sorted(PRE_DIR.glob("*.tif")):
        stem        = pre_path.stem   # bata-explosion_00000000_pre_disaster
        post_stem   = stem.replace("pre_disaster", "post_disaster")
        target_stem = stem.replace("pre_disaster", "building_damage")

        post_path   = POST_DIR   / f"{post_stem}.tif"
        target_path = TARGET_DIR / f"{target_stem}.tif"

        if not post_path.exists():
            print(f"  Skipping {stem} — post not found")
            continue

        pairs.append({
            "pre":    pre_path,
            "post":   post_path,
            "target": target_path if target_path.exists() else None,
            "stem":   stem
        })
    return pairs

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_batch():
    print("=" * 60)
    print("BRIGHT Dataset — Batch Damage Assessment")
    print("=" * 60)

    if not HAS_RASTERIO:
        print("\nERROR: rasterio is required for .tif files.")
        print("Run: pip install rasterio")
        return

    pairs = find_pairs()
    if not pairs:
        print("ERROR: No .tif pairs found.")
        return
    print(f"Found {len(pairs)} image pairs\n")

    model    = load_model()
    mask_gen = load_sam()
    all_rows = []

    for i, pair in enumerate(pairs):
        stem = pair["stem"]
        print(f"[{i+1}/{len(pairs)}] {stem}")

        pre_rgb  = load_tif(pair["pre"])
        post_rgb = load_tif(pair["post"])

        # Ensure both are 3-channel RGB uint8
        if pre_rgb.ndim == 2:
            pre_rgb = np.stack([pre_rgb]*3, axis=-1)
        if post_rgb.ndim == 2:
            post_rgb = np.stack([post_rgb]*3, axis=-1)

        # Match sizes
        if pre_rgb.shape != post_rgb.shape:
            post_rgb = cv2.resize(post_rgb,
                (pre_rgb.shape[1], pre_rgb.shape[0]),
                interpolation=cv2.INTER_AREA)

        gt_mask    = load_tif(pair["target"], is_target=True) if pair["target"] else None
        gt_overlay = render_gt_overlay(post_rgb, gt_mask)
        print(f"  GT: {'loaded' if gt_mask is not None else 'not found'} | "
              f"Image: {pre_rgb.shape[1]}x{pre_rgb.shape[0]}px")

        results      = classify_buildings(model, mask_gen, pre_rgb, post_rgb)
        pred_overlay = render_pred_overlay(post_rgb, results)

        dist = Counter(r["class_name"] for r in results)
        print(f"  Buildings: {len(results)} | " +
              " | ".join(f"{k}: {v}" for k,v in dist.most_common()))

        save_comparison(pre_rgb, post_rgb, gt_overlay, pred_overlay, results, stem)

        for r in results:
            all_rows.append({
                "image": stem,
                "building_id": len(all_rows),
                "predicted_class": r["class_name"],
                "confidence": round(r["confidence"],4),
                "score_no_damage":  round(r["scores"][0],4),
                "score_minor":      round(r["scores"][1],4),
                "score_major":      round(r["scores"][2],4),
                "score_destroyed":  round(r["scores"][3],4),
                "bbox_x1": r["bbox"][0], "bbox_y1": r["bbox"][1],
                "bbox_x2": r["bbox"][2], "bbox_y2": r["bbox"][3],
                "area_px": r["mask"]["area"],
                "has_gt": gt_mask is not None
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = OUTPUT_DIR / "damage_report_combined.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCSV: {csv_path}  ({len(df)} buildings total)")
        print("\nOverall prediction distribution:")
        print(df["predicted_class"].value_counts().to_string())

    print("\nDone.")

if __name__ == "__main__":
    run_batch()
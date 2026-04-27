# DAMAGESCOPE — Claude Code Context

## Project Overview

DAMAGESCOPE is a disaster damage assessment system that fuses **pre-event** and **post-event** GeoTIFF satellite imagery to detect and classify building/infrastructure damage after natural disasters using a **Siamese ResNet18** architecture.

The system consists of:
- **ML pipeline** — Siamese ResNet18 trained on BRIGHT dataset (PyTorch)
- **Backend** — FastAPI serving inference and precomputed results
- **Frontend** — Next.js dashboard with upload, live inference, map visualization, and precaution recommendations

> **SAM and Streamlit are completely removed from this project. Do not reference, import, or suggest them under any circumstance.**

---

## ⚠️ Current State — Read Before Writing a Single Line

| Component | Status | Notes |
|---|---|---|
| Siamese ResNet18 architecture | **Not built** | Build from scratch |
| Training script | **Not built** | Build from scratch |
| BRIGHT data pipeline | Exists partially | GeoTIFF pre/post/target folders present |
| FastAPI backend | **Not built** | Build from scratch |
| Next.js frontend | **Not built** | Build from scratch |
| Model weights | **Do not exist** | Nothing is trained yet |
| SAM | **Deleted — out of scope** | Never reference this |
| Streamlit | **Deleted — out of scope** | Never reference this |

---

## Repo Structure (Target)

```
damagescope/
├── data/
│   ├── pre-event/           # Pre-disaster GeoTIFF images
│   ├── post-event/          # Post-disaster GeoTIFF images
│   └── target/              # Ground truth damage masks (GeoTIFF)
├── ml/
│   ├── dataset.py           # PyTorch Dataset — GeoTIFF loader, pre/post pair matching
│   ├── model.py             # Siamese ResNet18 architecture
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation — IoU, F1, pixel accuracy
│   ├── inference.py         # Single image pair inference — returns damage mask
│   └── checkpoints/         # Saved model weights (gitignored)
├── backend/
│   ├── main.py              # FastAPI app entry
│   ├── routes/
│   │   ├── inference.py     # POST /inference — upload pre/post, return damage map
│   │   └── results.py       # GET /results — precomputed damage records
│   ├── schemas.py           # Pydantic models
│   └── utils.py             # GeoTIFF processing helpers
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Dashboard home
│   │   ├── upload/          # Upload pre/post images → run inference
│   │   ├── map/             # Map view of damage results
│   │   └── precautions/     # Precaution recommendations per damage class
│   ├── components/
│   └── ...
├── outputs/                 # Generated damage masks from inference
├── requirements.txt         # Python dependencies
├── .gitignore
└── CLAUDE.md
```

---

## Tech Stack

### ML
- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Model**: Siamese ResNet18 (custom — build from scratch using `torchvision.models.resnet18`)
- **Data format**: GeoTIFF — use `rasterio` for loading
- **Training hardware**: Local NVIDIA GPU — use `device = torch.device("cuda")`
- **Dataset**: BRIGHT (pre/post disaster GeoTIFF pairs + ground truth masks)

### Backend
- **Framework**: FastAPI
- **GeoTIFF processing**: `rasterio`, `numpy`
- **Model serving**: Load trained `.pth` weights, run inference per request
- **No database for now** — results stored as files in `outputs/`

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Map**: Leaflet.js or `react-leaflet` for damage overlay visualization
- **File upload**: Native HTML input or `react-dropzone`
- **Charts**: Recharts or Chart.js for dashboard stats

---

## ML — Siamese ResNet18 Architecture

### Concept
A Siamese network uses **two identical ResNet18 branches** (shared weights) to extract features from the pre-event and post-event images independently. The feature maps are subtracted (or concatenated) and passed to a classification head to predict damage class per pixel or per patch.

### Architecture Spec

```
Input: pre_image [B, C, H, W] + post_image [B, C, H, W]
         ↓                         ↓
   ResNet18 encoder           ResNet18 encoder   ← shared weights
         ↓                         ↓
   feature_pre               feature_post
         ↓
   diff = |feature_pre - feature_post|   (or torch.cat)
         ↓
   Classification head (Conv layers → damage class)
         ↓
   Output: damage_mask [B, num_classes, H, W]
```

### Damage Classes (BRIGHT dataset)
- 0: No Damage
- 1: Minor Damage
- 2: Major Damage
- 3: Destroyed

### Key Implementation Notes
- Use `torchvision.models.resnet18(pretrained=True)` as backbone — remove the final FC layer
- Shared weights = instantiate one ResNet18, pass both images through the same instance
- Loss: CrossEntropyLoss (multiclass segmentation) or BCEWithLogitsLoss (binary)
- Optimizer: AdamW, lr=1e-4
- Use mixed precision training: `torch.cuda.amp.autocast()`
- Save best checkpoint based on val IoU, not val loss

---

## Data Pipeline

### GeoTIFF Loading
- Use `rasterio` — NOT PIL or OpenCV (they don't handle GeoTIFF metadata correctly)
- GeoTIFF may have multiple bands (not just RGB) — check band count before assuming 3 channels
- Normalize per-band: `(image - mean) / std` using BRIGHT dataset statistics if available
- Pre/post pairs are matched by **filename** — `pre-event/scene_001.tif` matches `post-event/scene_001.tif`
- `target/` masks are single-band GeoTIFF with integer class values (0–3)

### Dataset Class Requirements
```python
# dataset.py must:
# 1. Scan pre-event/ and post-event/ and match pairs by filename
# 2. Load corresponding target/ mask
# 3. Apply same spatial augmentations to pre, post, AND mask together
# 4. Return: { 'pre': tensor, 'post': tensor, 'mask': tensor }
```

### Augmentations
- Apply to pre/post/mask together (use `albumentations` with `additional_targets`)
- Safe augmentations: RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90
- Do NOT apply color jitter independently to pre and post — it breaks the change signal

---

## Backend — FastAPI

### Routes to Build

```
POST /api/inference
  - Accepts: pre_image (file), post_image (file)
  - Loads model weights from checkpoints/best.pth
  - Runs inference, returns damage mask as PNG + damage class summary JSON

GET /api/results
  - Returns list of precomputed inference results from outputs/
  - Each result: { id, filename, damage_summary, timestamp, mask_url }

GET /api/precautions/{damage_class}
  - Returns precaution recommendations for given damage class
  - Hardcode initially, can be expanded later
```

### Key Rules
- Load model once at startup using FastAPI lifespan, not on every request
- Use `rasterio` for GeoTIFF processing in inference route
- Return damage mask as a standard PNG (convert from GeoTIFF after inference)
- All file paths from environment variables — never hardcode

---

## Frontend — Next.js

### Pages to Build

**`/` — Dashboard**
- Summary stats: total analyzed areas, damage distribution (pie/bar chart)
- Recent inference results list
- Quick links to Upload and Map

**`/upload` — Inference Upload**
- Drag-and-drop or file input for pre-event and post-event GeoTIFF
- Submit → POST `/api/inference` → show returned damage mask overlay
- Display damage class breakdown (% No Damage / Minor / Major / Destroyed)

**`/map` — Map Visualization**
- Leaflet map with damage mask overlay on satellite base layer
- Color-coded by damage class (green / yellow / orange / red)
- Click region → show damage class + precaution info

**`/precautions` — Precaution Dashboard**
- Per damage class: recommended actions, safety protocols, evacuation info
- Static content initially, driven by `/api/precautions/{damage_class}`

### Key Rules
- Use Next.js App Router — no Pages Router
- API calls go through `/app/api/` route handlers (proxy to FastAPI) — never call FastAPI directly from client components
- Use `react-leaflet` for map — do NOT use Google Maps (licensing)
- Tailwind CSS only — no other CSS frameworks
- Use `recharts` for dashboard charts

---

## Environment Variables

```env
# Backend
MODEL_WEIGHTS_PATH=./ml/checkpoints/best.pth
DATA_DIR=./data
OUTPUTS_DIR=./outputs

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Commands

```bash
# ML — train model (run after dataset is verified)
cd ml && python train.py

# ML — evaluate trained model
cd ml && python evaluate.py

# Backend — dev server
cd backend && uvicorn main:app --reload --port 8000

# Frontend — dev server
cd frontend && npm run dev
```

---

## Build Order — Follow This Strictly

Claude Code must build in this exact order. Do not skip ahead.

```
Step 1: ml/dataset.py
  → GeoTIFF loader, pair matching, augmentations
  → Verify data loads correctly before touching model code

Step 2: ml/model.py
  → Siamese ResNet18 architecture
  → Test forward pass with dummy tensors before training

Step 3: ml/train.py
  → Training loop with CUDA, mixed precision, checkpoint saving
  → Track: train loss, val loss, val IoU per epoch

Step 4: ml/evaluate.py
  → IoU, F1, pixel accuracy against target/ masks
  → Run after training completes

Step 5: ml/inference.py
  → Load best.pth, run on single pre/post pair, return damage mask

Step 6: backend/main.py + routes/
  → FastAPI with /inference and /results routes
  → Wire to ml/inference.py

Step 7: frontend/
  → Next.js pages in order: Dashboard → Upload → Map → Precautions
```

---

## What Claude Code Should / Should Not Do

### DO
- Follow the build order above — Step 1 before Step 2, always
- Use `rasterio` for all GeoTIFF operations
- Verify dataset loads correctly (print shapes, check band counts) before training
- Use shared ResNet18 weights in the Siamese network — one instance, two forward passes
- Save checkpoints as `best.pth` based on val IoU
- Use environment variables for all paths
- Test every module with a small dummy input before integrating

### DO NOT
- Import or reference SAM anywhere — it is gone
- Use Streamlit — it is gone
- Use PIL or OpenCV to load GeoTIFF files — use rasterio
- Apply augmentations to pre and post images independently — always transform together
- Start frontend before backend inference route works
- Start backend before ml/inference.py is verified
- Hardcode file paths, model paths, or API URLs

---

## Known Issues / Verify Before Starting

- [ ] Confirm band count of GeoTIFF files in `pre-event/` and `post-event/` (RGB = 3 bands, multispectral = more)
- [ ] Confirm `target/` mask value range — should be 0–3 for 4-class, or 0–1 for binary
- [ ] Confirm pre/post filenames match exactly across folders
- [ ] Verify NVIDIA GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check `requirements.txt` — add `rasterio`, `albumentations`, `fastapi`, `uvicorn` if missing

---

*Last updated: April 2026*
*Author: Tharun Kumar*
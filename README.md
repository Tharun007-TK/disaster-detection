# DAMAGESCOPE

DAMAGESCOPE detects and classifies building and infrastructure damage by comparing pre-event and post-event satellite GeoTIFF imagery. It produces per-pixel damage masks with four classes: 0 = No Damage, 1 = Minor, 2 = Major, 3 = Destroyed. The v2 stack comprises a Siamese ResNet-18 segmentation model (PyTorch), a FastAPI inference service, and a Next.js dashboard for visualization and uploads.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-stable-009688)
![Next.js](https://img.shields.io/badge/Next.js-16-000000)
![Azure](https://img.shields.io/badge/Azure-Container%20Apps-0078D4)

## Live URLs

| Component | URL |
|-----------|-----|
| Frontend  | https://black-beach-0144c460f.7.azurestaticapps.net |
| Backend   | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io |
| Health    | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io/api/health |

## Architecture

- Model: Siamese ResNet-18 (shared encoder weights). Two forward passes (pre/post), fusion via absolute difference |f_pre - f_post|, convolutional segmentation head outputting `[B, 4, H, W]`.
- Backend: FastAPI service using `rasterio` for GeoTIFF I/O, async inference via `asyncio.to_thread`, configurable eager loading for development.
- Frontend: Next.js 16 dashboard with Leaflet map overlay, drag-and-drop GeoTIFF upload, and inference result charts.

Simple diagram:

```
pre.tif  --> encoder ---\
                             diff --> conv head --> mask [B,4,H,W]
post.tif --> encoder ---/

mask -> served by FastAPI -> visualized on Next.js + Leaflet
```

## Dataset

This project is trained on the BRIGHT dataset (pre-event optical RGB + post-event SAR GeoTIFF pairs) with pixel-level damage labels.

Expected dataset layout (local):

```
data/
  pre-event/    # pre-event GeoTIFFs (e.g. scene_0001.tif)
  post-event/   # post-event GeoTIFFs with matching filenames
  target/       # single-band integer GeoTIFF masks (values 0..3)
```

Notes:
- Filenames must match between `pre-event/` and `post-event/` to form pairs.
- Use `rasterio` for reading/writing GeoTIFF to preserve metadata.
- Spatial augmentations in training must be applied jointly to pre/post/mask.

## Local setup

Requirements: Python 3.11, Node 18+, Docker (for container builds). GPU optional for training.

1. Clone and create a venv

```bash
git clone https://github.com/Tharun007-TK/disaster-detection.git
cd disaster-detection
python -m venv venv
venv\Scripts\activate   # Windows
```

2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
```

3. Run backend (development)

```bash
cd backend
# optionally: set EAGER_MODEL_LOAD=1 to force eager load in dev
set EAGER_MODEL_LOAD=1
uvicorn main:app --reload --port 8000
```

4. Run frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

5. Example inference request

```bash
curl -X POST "http://localhost:8000/api/inference" -F "pre=@/path/to/pre.tif" -F "post=@/path/to/post.tif"
```

## ML model & training

Key implementation notes:

- Architecture: Siamese ResNet-18 with shared encoder. Feature fusion via `abs(f_pre - f_post)`, followed by convolutional segmentation head producing 4-class logits.
- Loss: `CrossEntropyLoss` (multiclass segmentation).
- Optimizer: `AdamW`.
- Scheduler: `CosineAnnealing`.
- Mixed precision: AMP (`torch.cuda.amp`) used during training.
- Checkpoint: `ml/checkpoints/best.pth` (saved at peak validation IoU, ~49 MB).

Training entrypoint and helpers are under `ml/`.

Run training (example):

```bash
cd ml
python train.py --config configs/train.yaml
```

Hyperparameters (defaults):

| Parameter      | Default |
|----------------|---------|
| Loss           | CrossEntropyLoss |
| Optimizer      | AdamW   |
| LR             | 1e-4    |
| Scheduler      | CosineAnnealing |
| Mixed precision| AMP     |
| Batch size     | TODO    |
| Epochs         | TODO    |

## Model metrics

Evaluation summary (placeholders — update after evaluation):

| Metric         | Value |
|----------------|-------|
| Pixel Accuracy | TODO  |
| Mean IoU       | TODO  |
| Macro F1       | TODO  |

## Backend API

Primary routes (FastAPI):

- `POST /api/inference`            : batch inference endpoint accepting pre/post GeoTIFFs
- `POST /api/inference/single`     : single-pair inference helper
- `GET  /api/health`               : health check
- `GET  /api/results`              : list precomputed outputs in `backend/outputs/`
- `GET  /api/precautions/{class}`  : returns precaution recommendations for a damage class

Implementation notes:

- GeoTIFF I/O uses `rasterio` to preserve CRS and transforms.
- Model loading controlled by `MODEL_WEIGHTS_PATH` env var (default `./ml/checkpoints/best.pth`).
- Async inference uses `asyncio.to_thread` to run CPU/GPU-bound work without blocking the event loop.
- Docker image is CPU-only and optimized for size (~700 MB).

## Deployment

- Backend: Azure Container Apps (2 CPU / 4 Gi). Frontend: Azure Static Web Apps (Free tier).
- CI/CD: GitHub Actions workflows under `.github/workflows/` (`azure-backend.yml`, `azure-frontend.yml`).
- See `docs/PROJECT_REPORT.md` for full deployment steps and architecture diagrams.

## Repository structure

```
.
├── ml/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── Dockerfile
│   └── outputs/
├── frontend/
├── data/         # pre-event/, post-event/, target/
├── docs/
└── .github/workflows/
```

## Tech stack

Python 3.11, PyTorch 2.6, FastAPI, Next.js 16 (TypeScript), Leaflet, `rasterio`, `albumentations` (training), Docker, Azure Container Apps, Azure Static Web Apps, GitHub Actions.

## License

MIT License

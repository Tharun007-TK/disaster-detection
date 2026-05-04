# DAMAGESCOPE — Project Report

**Disaster Damage Assessment via Siamese ResNet-18 on Pre/Post-Event Satellite Imagery**

Author: Tharun Kumar (727622BAM046@mcet.in)
Last updated: 2026-05-04

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset — BRIGHT](#3-dataset--bright)
4. [Model — Siamese ResNet-18](#4-model--siamese-resnet-18)
5. [Training Pipeline](#5-training-pipeline)
6. [Backend — FastAPI](#6-backend--fastapi)
7. [Frontend — Next.js Dashboard](#7-frontend--nextjs-dashboard)
8. [Bugs Encountered and Fixes](#8-bugs-encountered-and-fixes)
9. [Azure Deployment](#9-azure-deployment)
10. [GitHub Actions CI/CD](#10-github-actions-cicd)
11. [Verification and Monitoring](#11-verification-and-monitoring)
12. [Cost Analysis](#12-cost-analysis)
13. [Appendix](#13-appendix)

---

## 1. Project Overview

DAMAGESCOPE detects and classifies building/infrastructure damage after natural disasters by comparing pre-event and post-event satellite imagery (GeoTIFF). A Siamese ResNet-18 network outputs a per-pixel damage class:

| Class ID | Label        | Color   |
|----------|--------------|---------|
| 0        | No Damage    | Green   |
| 1        | Minor Damage | Yellow  |
| 2        | Major Damage | Orange  |
| 3        | Destroyed    | Red     |

The system has three deployable parts:

- **ML pipeline** — PyTorch training + inference on the BRIGHT dataset
- **Backend** — FastAPI inference service (CPU-only runtime image)
- **Frontend** — Next.js 16 dashboard with upload, map view, and precaution recommendations

> Place screenshot: `docs/images/01_dashboard.png` — final frontend dashboard

![Dashboard](images/01_dashboard.png)

---

## 2. System Architecture

```
┌──────────────────────┐   POST /api/inference    ┌──────────────────────┐
│  Next.js Frontend    │ ───────────────────────▶ │  FastAPI Backend     │
│  (Static Web Apps)   │                          │  (Container Apps)    │
│                      │ ◀───────────────────────│                      │
│  - Upload            │   damage mask + JSON     │  - Lazy model load   │
│  - Map view          │                          │  - rasterio I/O      │
│  - Precautions       │                          │  - Siamese ResNet18  │
└──────────────────────┘                          └──────────────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │  ml/checkpoints/     │
                                                  │  best.pth (49 MB)    │
                                                  └──────────────────────┘
```

> Place screenshot: `docs/images/02_architecture.png` — high-level cloud diagram

![Architecture](images/02_architecture.png)

### Data Flow

1. User uploads a `pre.tif` and `post.tif` pair via the dashboard.
2. Browser POSTs `multipart/form-data` directly to the backend FQDN (bypasses any 4.5 MB serverless function body limit).
3. Backend runs Siamese inference on CPU, writes the mask + summary JSON to `OUTPUTS_DIR`, and returns a JSON payload.
4. Frontend renders the mask overlay and damage breakdown chart.

---

## 3. Dataset — BRIGHT

The **B**uilding damage assessment via **R**emote sensing for **I**nternational humanitarian **GH**T (BRIGHT) dataset is used as the training source.

### File Layout

```
data/
├── pre-event/
│   ├── beirut-explosion_00000014_pre_disaster.tif
│   ├── haiti-earthquake_00000042_pre_disaster.tif
│   └── ...
├── post-event/
│   ├── beirut-explosion_00000014_post_disaster.tif
│   ├── haiti-earthquake_00000042_post_disaster.tif
│   └── ...
└── target/
    ├── beirut-explosion_00000014_building_damage.tif
    ├── haiti-earthquake_00000042_building_damage.tif
    └── ...
```

| Folder        | Bands | dtype  | Notes                                    |
|---------------|-------|--------|------------------------------------------|
| `pre-event/`  | 3     | uint8  | RGB optical                              |
| `post-event/` | 1     | uint8  | SAR (synthetic-aperture radar)           |
| `target/`     | 1     | uint8  | Pixel-level damage class (0–3)           |

Pairs are matched by **filename prefix** (`<event>_<id>`).

### Loading

GeoTIFFs require `rasterio` — PIL/OpenCV silently corrupt georeferenced metadata and multi-band SAR imagery.

```python
# ml/dataset.py
import rasterio

def _read_geotiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()  # shape: (bands, H, W)
    return arr
```

> Place screenshot: `docs/images/03_dataset_sample.png` — pre/post/target triplet visualization

![Dataset sample](images/03_dataset_sample.png)

### Augmentation

`albumentations` is used during **training only** — kept out of the runtime image. Spatial transforms are applied jointly to pre, post, and mask via `additional_targets`:

```python
A.Compose([
    A.RandomCrop(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
], additional_targets={"post": "image", "mask": "mask"})
```

Color jitter is **never** applied independently to pre vs post — that destroys the change signal the network is supposed to learn.

---

## 4. Model — Siamese ResNet-18

### Architecture

```
Input:  pre  [B, 3, H, W]            post [B, 1→3, H, W]
              │                            │
              ▼                            ▼
        ┌──────────────┐             ┌──────────────┐
        │  ResNet-18   │   shared    │  ResNet-18   │
        │  (encoder)   │ ◀── weights ▶│  (encoder)   │
        └──────────────┘             └──────────────┘
              │                            │
              └────── feature_pre ─────────┘
                          │
                  diff = |f_pre − f_post|
                          │
                          ▼
                  ┌──────────────────┐
                  │  Conv head       │
                  │  (4-class seg)   │
                  └──────────────────┘
                          │
                          ▼
                Output: mask [B, 4, H, W]
```

### Key Choices

- **Shared weights**: one ResNet-18 instance, two forward passes — guarantees the encoder learns features that are *comparable* between pre and post.
- **Backbone init**: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` — ImageNet-pretrained, FC layer stripped.
- **Feature fusion**: absolute difference, NOT concatenation. Difference is invariant to which image is "pre" — useful when test-time pre/post is ambiguous.
- **Loss**: `CrossEntropyLoss` over 4 classes.
- **Optimizer**: `AdamW`, lr=1e-4, weight_decay=1e-4.
- **Mixed precision**: `torch.cuda.amp.autocast()` for ~2× training throughput on GPU.
- **Checkpoint policy**: save best by **validation IoU**, not val loss (loss can decrease while IoU plateaus).

> Place screenshot: `docs/images/04_model_diagram.png` — architecture diagram

![Model](images/04_model_diagram.png)

---

## 5. Training Pipeline

### Hardware

Local NVIDIA GPU. CUDA detected via:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Hyperparameters

| Parameter      | Value           |
|----------------|-----------------|
| Batch size     | 8               |
| Image size     | 512 × 512       |
| Epochs         | 50              |
| Optimizer      | AdamW           |
| Learning rate  | 1e-4            |
| Scheduler      | CosineAnnealing |
| Mixed precision| Enabled (AMP)   |

### Training Curves

Training and validation IoU/F1 across epochs:

> Place screenshot: `docs/images/05_training_curves.png` — loss + IoU curves

![Training curves](images/05_training_curves.png)

### Final Metrics

Model `ml/checkpoints/best.pth` (49 MB) is the snapshot at the epoch with peak validation IoU.

| Metric           | Value |
|------------------|-------|
| Pixel accuracy   | _fill in from `eval_report.json`_ |
| Mean IoU         | _fill in_ |
| Macro F1         | _fill in_ |

> Place screenshot: `docs/images/06_confusion_matrix.png` — confusion matrix from `ml/checkpoints/confusion_matrix.png`

![Confusion matrix](images/06_confusion_matrix.png)

---

## 6. Backend — FastAPI

### Routes

| Method | Path                          | Description                         |
|--------|-------------------------------|-------------------------------------|
| GET    | `/`                           | Service banner                      |
| GET    | `/api/health`                 | Health probe (`status`, `model_loaded`) |
| POST   | `/api/inference`              | Pre/post pair inference             |
| POST   | `/api/inference/single`       | Single-image inference              |
| GET    | `/api/results`                | List precomputed results            |
| GET    | `/api/precautions/{class}`    | Per-class precaution recommendations|

### Lazy Model Loading

The model is **not** loaded at startup. Reason: the boot RAM spike (uvicorn worker baseline + torch import + model weights all at once) was tripping Render's free-tier 512 MB OOM killer.

```python
# backend/state.py
def get_model() -> tuple[SiameseDamageNet, torch.device]:
    global model, device
    if model is not None:
        return model, device
    with _lock:
        if model is None:
            from ml.inference import load_model
            ckpt = _resolve_ckpt_path()
            print(f"[state] lazy-loading model from {ckpt}")
            m, d = load_model(ckpt)
            m.eval()
            model, device = m, d
    return model, device
```

Set `EAGER_MODEL_LOAD=1` to opt back into eager startup for local dev.

### CORS

Env-var driven:
```python
_cors_env = os.environ.get("CORS_ORIGINS", "").strip()
_allow_origins = [o.strip() for o in _cors_env.split(",")] if _cors_env else ["*"]
```

In production the SWA URL is set: `CORS_ORIGINS=https://black-beach-0144c460f.7.azurestaticapps.net`.

### Async Inference

GeoTIFF read + torch forward are blocking. Run them off the event loop:
```python
result = await asyncio.to_thread(
    run_inference, pre_path, post_path,
    ckpt_path=_ckpt_path(), out_dir=out_dir,
    model=model, device=device,
)
```

---

## 7. Frontend — Next.js Dashboard

### Pages

| Route          | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `/`            | Dashboard summary + recent results                         |
| `/upload`      | Drag/drop pre + post GeoTIFF, run inference                |
| `/map`         | Leaflet map with damage mask overlay + class color coding |
| `/precautions` | Per-class safety + evacuation guidance                     |

### API URL Wiring

`process.env.NEXT_PUBLIC_API_URL` baked into client bundle at **build time**:
```ts
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
```

This is critical — `NEXT_PUBLIC_*` vars must be present *during* `next build`, not at runtime. SWA `appsettings` (runtime) cannot influence the client bundle.

> Place screenshot: `docs/images/07_upload.png` — upload UI

![Upload UI](images/07_upload.png)

> Place screenshot: `docs/images/08_map.png` — Leaflet damage overlay

![Map view](images/08_map.png)

---

## 8. Bugs Encountered and Fixes

A chronological log of the issues hit during development and deployment, plus the resolution applied. Useful both as a postmortem and as a reference for future contributors.

### 8.1 Render Free-Tier OOM on Cold Boot

**Symptom**: backend died on startup with no traceback.
**Root cause**: torch + model weights + uvicorn worker baseline exceeded 512 MB at the same instant.
**Fix**: lazy model load gated by `EAGER_MODEL_LOAD` env. Commit `05ff0cc`.

### 8.2 Vercel Function Body Limit (4.5 MB)

**Symptom**: 413 Payload Too Large on uploads.
**Root cause**: GeoTIFFs > 4.5 MB hit Vercel's serverless function body limit.
**Fix**: frontend POSTs directly to backend FQDN, bypassing the Next.js `/api/*` proxy. Commit `21dea96`.

### 8.3 CORS Block from Vercel Preview Domains

**Symptom**: browser blocked cross-origin upload from `*.vercel.app`.
**Fix**: backend CORS now reads `CORS_ORIGINS` env + has a regex fallback for `*.vercel.app`. Commit `a1e25b1`.

### 8.4 PowerShell vs Bash Syntax

**Symptom**:
```
RG=damagescope-rg : The term 'RG=damagescope-rg' is not recognized…
```
**Root cause**: bash variable assignment syntax run in PowerShell.
**Fix**: PowerShell uses `$VAR = "value"`, backtick `` ` `` for line continuation, `$ENV` is reserved (use `$ENV_NAME`).

### 8.5 Azure Container App Cannot Pull Image (UNAUTHORIZED)

**Symptom**: workflow built + pushed image successfully, but Container App revision failed to provision:
```
Field 'template.containers.***.image' is invalid with details:
'Invalid value: ".azurecr.io/damagescope-backend:<sha>":
GET https:?scope=repository%3Adamagescope-backend%3Apull&service=***.azurecr.io:
UNAUTHORIZED: authentication required'
```
**Root cause**: Container App had no registered ACR pull credentials.
**Fix**: register ACR admin user with the Container App once:
```powershell
$ACR = az acr list -g damagescope-rg --query "[0].name" -o tsv
$ACR_USER = az acr credential show -n $ACR --query username -o tsv
$ACR_PASS = az acr credential show -n $ACR --query "passwords[0].value" -o tsv
az containerapp registry set -n damagescope-api -g damagescope-rg `
  --server "$ACR.azurecr.io" --username $ACR_USER --password $ACR_PASS
```
Long-term: switch to managed identity + `AcrPull` role (see Appendix).

### 8.6 `albumentations` ModuleNotFoundError on Inference

**Symptom**: 500 on first inference call:
```
File "/app/ml/dataset.py", line 14, in <module>
    import albumentations as A
ModuleNotFoundError: No module named 'albumentations'
```
**Root cause**: `albumentations` was correctly removed from runtime requirements but `ml/dataset.py` still had a top-level import. Inference imports `NUM_CLASSES` from `ml.dataset`, which triggered the chain.
**Fix**: move `albumentations` import behind `TYPE_CHECKING` and into the training-only transform builders:
```python
if TYPE_CHECKING:
    import albumentations as A

def build_train_transform(img_size: int = 512) -> "A.Compose":
    import albumentations as A  # training-only dep
    ...
```

### 8.7 Invalid Service Principal Secret

**Symptom**: `AZURE_CREDENTIALS` rejected at `azure/login@v2`:
```
AADSTS7000215: Invalid client secret provided.
Ensure the secret being sent in the request is the client secret value,
not the client secret ID, for a secret added to app '***'.
```
**Root cause**: secret was malformed when copied into GitHub repo secrets.
**Fix**: regenerate with `--sdk-auth` and paste the **entire JSON object** (including `clientId`, `clientSecret`, `tenantId`, `subscriptionId`):
```powershell
az ad sp create-for-rbac --name "damagescope-github" `
  --role contributor `
  --scopes "/subscriptions/$SUB/resourceGroups/$RG" `
  --sdk-auth | Out-File "$HOME\azure-creds.json" -Encoding utf8
```

### 8.8 `NEXT_PUBLIC_API_URL` Returned `null` on SWA

**Symptom**: dashboard hit `localhost:8000` despite SWA `appsettings` containing the right value.
**Root cause**: `NEXT_PUBLIC_*` is **build-time** in Next.js — SWA `appsettings` are only injected at runtime to Functions, not the client bundle.
**Fix**: pass via the workflow `env:` block on the build step:
```yaml
- name: Build and deploy
  uses: Azure/static-web-apps-deploy@v1
  env:
    NEXT_PUBLIC_API_URL: ${{ secrets.NEXT_PUBLIC_API_URL }}
  with:
    azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
    ...
```

### 8.9 Multi-line YAML `|` Mangled Env Vars

**Symptom**: 503 on inference with body:
```json
{
  "detail": "Model load failed: [Errno 2] No such file or directory:
   '/app/ml/checkpoints/best.pth\nOUTPUTS_DIR=/app/backend/outputs\n
    EAGER_MODEL_LOAD=0\nCORS_ORIGINS=https:/black-beach-...'"
}
```
**Root cause**: `azure/container-apps-deploy-action@v2` expects `environmentVariables` as a **space-separated single line**. The workflow used YAML literal `|` which preserves newlines — so the action passed the entire blob as one giant value of the first env var.
**Fix**: use a single-line space-separated string:
```yaml
environmentVariables: MODEL_WEIGHTS_PATH=/app/ml/checkpoints/best.pth OUTPUTS_DIR=/app/backend/outputs EAGER_MODEL_LOAD=0 CORS_ORIGINS=https://black-beach-0144c460f.7.azurestaticapps.net
```

### 8.10 Git Push Rejected (non-fast-forward)

**Symptom**:
```
! [rejected]        main -> main (non-fast-forward)
Updates were rejected because the tip of your current branch is behind
its remote counterpart.
```
**Root cause**: an `imgbot` PR was merged remotely while local had unpushed work.
**Fix**: rebase local onto remote, push:
```powershell
git stash push -- ml/dataset.py
git pull --rebase origin main
git stash pop
git push origin main
```

---

## 9. Azure Deployment

### 9.1 Subscription

Used the **Azure for Students** plan ($100 credit, no card required). Verify:
```powershell
az account show --query name
```
Should print `Azure for Students`.

### 9.2 Resource Inventory

| Resource                       | Service                  | SKU         | Purpose                          |
|--------------------------------|--------------------------|-------------|----------------------------------|
| `damagescope-rg`               | Resource Group           | —           | Container for everything         |
| `damagescopeacr<rand>`         | Container Registry       | Basic       | Docker image hosting             |
| `damagescope-env`              | Container Apps Env       | Consumption | Managed env (Log Analytics auto) |
| `damagescope-api`              | Container App            | 2 CPU / 4 Gi| FastAPI runtime                  |
| `damagescope-web`              | Static Web App           | Free        | Next.js frontend                 |
| `damagescope-github` (SP)      | Service Principal        | —           | GitHub Actions auth              |

### 9.3 Setup Commands (PowerShell)

```powershell
# vars
$RG = "damagescope-rg"
$LOCATION = "eastus"
$ACR = "damagescopeacr$(Get-Random -Maximum 9999)"
$ENV_NAME = "damagescope-env"
$APP = "damagescope-api"

# resource group
az group create -n $RG -l $LOCATION

# container registry (admin enabled for simplicity)
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true

# container apps extension + providers
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# container apps environment (provisions Log Analytics workspace, ~3 min)
az containerapp env create -n $ENV_NAME -g $RG -l $LOCATION

# placeholder container app (workflow updates the image later)
az containerapp create -n $APP -g $RG --environment $ENV_NAME `
  --image mcr.microsoft.com/k8se/quickstart:latest `
  --target-port 8000 --ingress external `
  --cpu 2.0 --memory 4.0Gi `
  --min-replicas 1 --max-replicas 2

# wire ACR pull credentials into the Container App
$ACR_USER = az acr credential show -n $ACR --query username -o tsv
$ACR_PASS = az acr credential show -n $ACR --query "passwords[0].value" -o tsv
az containerapp registry set -n $APP -g $RG `
  --server "$ACR.azurecr.io" --username $ACR_USER --password $ACR_PASS

# service principal for GitHub Actions
$SUB = az account show --query id -o tsv
az ad sp create-for-rbac --name "damagescope-github" `
  --role contributor `
  --scopes "/subscriptions/$SUB/resourceGroups/$RG" `
  --sdk-auth | Out-File "$HOME\azure-creds.json" -Encoding utf8

# static web app
az staticwebapp create -n damagescope-web -g $RG -l eastus2 --sku Free
az staticwebapp secrets list -n damagescope-web -g $RG `
  --query "properties.apiKey" -o tsv
```

### 9.4 Live URLs

| Component | URL                                                                  |
|-----------|----------------------------------------------------------------------|
| Frontend  | https://black-beach-0144c460f.7.azurestaticapps.net                   |
| Backend   | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io |
| Health    | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io/api/health |

> Place screenshot: `docs/images/09_azure_portal.png` — Container App overview in Azure portal

![Azure portal](images/09_azure_portal.png)

> Place screenshot: `docs/images/10_swa_overview.png` — SWA overview

![SWA overview](images/10_swa_overview.png)

---

## 10. GitHub Actions CI/CD

### 10.1 Required Repository Secrets

Settings → Secrets and variables → Actions → New repository secret:

| Name                              | Value                                                              |
|-----------------------------------|--------------------------------------------------------------------|
| `AZURE_CREDENTIALS`               | Full JSON from `az ad sp create-for-rbac --sdk-auth`               |
| `AZURE_RG`                        | `damagescope-rg`                                                   |
| `AZURE_ACR_NAME`                  | The `$ACR` name picked above                                       |
| `AZURE_CONTAINERAPP_NAME`         | `damagescope-api`                                                  |
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | tsv from `az staticwebapp secrets list`                            |
| `NEXT_PUBLIC_API_URL`             | `https://damagescope-api.<env>.eastus.azurecontainerapps.io`       |

### 10.2 Backend Workflow

`.github/workflows/azure-backend.yml`:

```yaml
name: Deploy backend to Azure Container Apps

on:
  push:
    branches: [main]
    paths:
      - "backend/**"
      - "ml/**"
      - ".github/workflows/azure-backend.yml"
      - "backend/Dockerfile"
      - ".dockerignore"
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
          auth-type: SERVICE_PRINCIPAL

      - uses: azure/container-apps-deploy-action@v2
        with:
          appSourcePath: ${{ github.workspace }}
          dockerfilePath: backend/Dockerfile
          acrName: ${{ secrets.AZURE_ACR_NAME }}
          containerAppName: ${{ secrets.AZURE_CONTAINERAPP_NAME }}
          resourceGroup: ${{ secrets.AZURE_RG }}
          imageToBuild: ${{ secrets.AZURE_ACR_NAME }}.azurecr.io/damagescope-backend:${{ github.sha }}
          targetPort: 8000
          ingress: external
          environmentVariables: MODEL_WEIGHTS_PATH=/app/ml/checkpoints/best.pth OUTPUTS_DIR=/app/backend/outputs EAGER_MODEL_LOAD=0 CORS_ORIGINS=https://black-beach-0144c460f.7.azurestaticapps.net
```

### 10.3 Frontend Workflow

`.github/workflows/azure-frontend.yml`:

```yaml
name: Deploy frontend to Azure Static Web Apps

on:
  push:
    branches: [main]
    paths: ["frontend/**", ".github/workflows/azure-frontend.yml"]
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches: [main]
    paths: ["frontend/**"]

jobs:
  build_and_deploy:
    if: github.event_name == 'push' || (github.event_name == 'pull_request' && github.event.action != 'closed')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - uses: Azure/static-web-apps-deploy@v1
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.NEXT_PUBLIC_API_URL }}
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: upload
          app_location: "frontend"
          api_location: ""
          output_location: ".next"

  close_pr:
    if: github.event_name == 'pull_request' && github.event.action == 'closed'
    runs-on: ubuntu-latest
    steps:
      - uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          action: close
```

### 10.4 Backend Dockerfile

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY ml /app/ml

ENV MODEL_WEIGHTS_PATH=/app/ml/checkpoints/best.pth \
    OUTPUTS_DIR=/app/backend/outputs \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
```

CPU-only torch wheels keep the image at ~700 MB (vs ~2 GB CUDA build).

> Place screenshot: `docs/images/11_actions_success.png` — green workflow run on the Actions tab

![Actions success](images/11_actions_success.png)

---

## 11. Verification and Monitoring

### 11.1 Health Probe

```powershell
curl https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io/api/health
```
Expected:
```json
{ "status": "ok", "model_loaded": false }
```

After the first inference call, `model_loaded` flips to `true` for the lifetime of the replica.

### 11.2 Live Logs

```powershell
az containerapp logs show -n damagescope-api -g damagescope-rg --type console --follow
```

Successful inference trace:
```
[state] lazy-loading model from /app/ml/checkpoints/best.pth
[state] model ready on cpu
pair inference start: pre=beirut-explosion_00000014_pre_disaster.tif (...)
pair inference done: <id>
INFO: ... "POST /api/inference HTTP/1.1" 200 OK
```

### 11.3 Revision Status

```powershell
az containerapp revision list -n damagescope-api -g damagescope-rg `
  --query "[?properties.active].{name:name, replicas:properties.replicas, healthState:properties.healthState, runningState:properties.runningState}" `
  -o table
```

Want: `Healthy` + `Running` + at least 1 replica.

### 11.4 Sample Inference Result

> Place screenshot: `docs/images/12_inference_result.png` — frontend showing damage mask overlay + pie chart

![Inference result](images/12_inference_result.png)

---

## 12. Cost Analysis

Burn under **Azure for Students** ($100 credit):

| Resource              | Tier             | Monthly cost (USD) |
|-----------------------|------------------|--------------------|
| Container Apps        | 2 CPU / 4 Gi, 1 min replica | ~$25–35  |
| Container Registry    | Basic            | ~$5                |
| Static Web Apps       | Free             | $0                 |
| Log Analytics         | Pay-as-you-go    | <$1 (low ingestion)|
| Outbound bandwidth    | First 100 GB free| ~$0                |
| **Total**             |                  | **~$30–40 / month**|

At ~$35/month, $100 student credit lasts roughly **2.8 months** of always-warm operation. Strategies to extend runway:

- Set `--min-replicas 0` (scale-to-zero) — cuts idle cost to ~$0, adds ~30–60 s cold-start delay.
- Drop to `1.0 CPU / 2.0 Gi` (works but inference is slower; risk of OOM during peak).
- Pause the resource group when not demoing: `az group delete -n damagescope-rg --yes` (destructive — back up `best.pth` and outputs first).

---

## 13. Appendix

### 13.1 Switch to Managed Identity (production-grade)

Admin user on ACR is convenient but anti-pattern. Replace with a system-assigned managed identity:

```powershell
$APP = "damagescope-api"
$RG = "damagescope-rg"
$ACR = "<acr-name>"

az containerapp identity assign -n $APP -g $RG --system-assigned
$MI = az containerapp identity show -n $APP -g $RG --query principalId -o tsv
$ACR_ID = az acr show -n $ACR --query id -o tsv

az role assignment create --assignee $MI --role AcrPull --scope $ACR_ID
az containerapp registry set -n $APP -g $RG `
  --server "$ACR.azurecr.io" --identity system

az acr update -n $ACR --admin-enabled false
```

### 13.2 `.dockerignore`

```
**/__pycache__
**/*.pyc
**/.pytest_cache
**/node_modules
.git
.github
frontend
data
backend/outputs/*
!backend/outputs/.gitkeep
ml/checkpoints/last.pth
ml/checkpoints_weighted
ml/checkpoints/samples
ml/checkpoints/train.log
ml/checkpoints/eval_report.json
ml/checkpoints/metrics.json
ml/checkpoints/confusion_matrix.png
*.md
.env
.env.*
.vscode
.idea
```

### 13.3 `.gitignore` Additions for Cloud Credentials

```
# Azure / cloud credentials — never commit
azure_creds.json
azure-creds.json
*.azureauth
.azure/
```

### 13.4 Commands Cheat Sheet

| Task                              | Command                                                                                        |
|-----------------------------------|------------------------------------------------------------------------------------------------|
| Tail backend logs                 | `az containerapp logs show -n damagescope-api -g damagescope-rg --type console --follow`       |
| Tail orchestration events         | `az containerapp logs show -n damagescope-api -g damagescope-rg --type system --tail 50`       |
| Force-restart backend revision    | `az containerapp revision restart -n damagescope-api -g damagescope-rg --revision <name>`      |
| Update env vars (replace all)     | `az containerapp update -n damagescope-api -g damagescope-rg --replace-env-vars KEY=VAL …`     |
| Update env vars (additive)        | `az containerapp update -n damagescope-api -g damagescope-rg --set-env-vars KEY=VAL …`         |
| Show current image                | `az containerapp show -n damagescope-api -g damagescope-rg --query "properties.template.containers[0].image" -o tsv` |
| Trigger backend workflow manually | `gh workflow run azure-backend.yml --ref main`                                                 |
| Trigger frontend workflow manually| `gh workflow run azure-frontend.yml --ref main`                                                |

### 13.5 Files Created During This Project

```
backend/Dockerfile                    # CPU-only torch image
.dockerignore                         # Strip junk from build context
.github/workflows/azure-backend.yml   # Container Apps deploy
.github/workflows/azure-frontend.yml  # Static Web Apps deploy
docs/PROJECT_REPORT.md                # This document
docs/images/                          # Screenshot directory
```

### 13.6 Future Work

- Switch ACR auth to managed identity (Appendix 13.1).
- Move outputs to Azure Blob Storage (current `backend/outputs/` is wiped on every revision rollover).
- Add Application Insights for structured tracing.
- Add a smoke test in CI that posts a known-good GeoTIFF pair to the staging revision and asserts a 200 response before flipping traffic.
- Train on a larger BRIGHT subset and report updated metrics.

---

*End of report.*

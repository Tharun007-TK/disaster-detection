# DAMAGESCOPE — Disaster Damage Assessment

> **Full project report:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) — dataset, model, deployment to Azure, bug log, and CI/CD.
>
> **Note:** the legacy README content below references SAM and Streamlit, both removed from the v2 architecture. See the report for the current Siamese ResNet-18 + FastAPI + Next.js stack.

![Status](https://img.shields.io/badge/Status-Deployed-green)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)
![Next.js](https://img.shields.io/badge/Next.js-16-000000)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CPU-orange)
![Azure](https://img.shields.io/badge/Azure-Container%20Apps%20%2B%20SWA-0078D4)

## Live URLs

| Component | URL |
|-----------|-----|
| Frontend  | https://black-beach-0144c460f.7.azurestaticapps.net |
| Backend   | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io |
| Health    | https://damagescope-api.delightfulrock-79f11601.eastus.azurecontainerapps.io/api/health |

---

## Legacy README (v1, outdated)

![Damage Assessment System](https://img.shields.io/badge/Status-Prototype-blue) 
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)

A multimodal pipeline designed to assess building damage by combining pre-disaster **Optical (Sentinel-2 L2A)** imagery and post-disaster **SAR (Sentinel-1 GRD)** imagery. The system utilizes Meta's **Segment Anything Model (SAM)** for zero-shot building footprint extraction and a custom ResNet-based dual-encoder network for damage classification.

## ✨ Features

- **Multimodal Inference**: Processes multi-band TIFFs (Optical RGB + SAR backscatter).
- **SAM Segmentation**: Leverages SAM (`vit_b`) to automatically segment building boundaries without requiring pre-trained geographic building footprint models.
- **Damage Classification**: Classifies structures into four severity levels aligned with the project schema:
  - `No Damage`
  - `Minor Damage` 
  - `Major Damage`
  - `Destroyed`
- **Batch Processing**: Rapidly processes entire directories of pre/post/target TIFF pairs, outputting consolidated CSV reports and visual overlays.
- **Interactive Web UI**: A beautiful, dark-themed Streamlit application enabling single-inference file uploads or massive batch runs with real-time analytics.

---

## 🛠️ Installation & Setup

### 1. Clone & Environment Setup
Clone the repository and install the required dependencies inside a virtual environment.
```bash
git clone https://github.com/your-username/disaster-damage-assessment.git
cd disaster-damage-assessment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note**: For Windows users or those processing `.tif` files, the `rasterio` package is strictly required.

### 2. Download SAM Weights
The pipeline relies on the Segment Anything Model (`vit_b`). Download the checkpoint to the `models/` directory:
```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```
*(File size is ~357 MB)*

---

## 🚀 Usage

You can interact with the pipeline either through the **Command Line Interface (CLI)** or the **Interactive Streamlit Web App**.

### Web Application (Recommended)
Run the Streamlit application for a full GUI with imagery uploads, prediction overlays, and data tables.
```bash
streamlit run app.py
```
- Navigate to `http://localhost:8501`.
- Choose **Single Inference** to upload your own Pre-event `.tif`, Post-event `.tif`, and (optionally) Target mask `.tif`.
- Choose **Batch Run** to automatically process the dataset in the `data/` folder.

### Command Line / Batch Mode
You can invoke the backend prototype script directly. Out of the box, it will search the `data/pre-event/` and `data/post-event/` directories to find matching TIF pairs and evaluate them en masse.
```bash
python damage_assessment_prototype.py
```
Output overlays and CSV reports will be saved to the `outputs/` directory.

---

## 📂 Project Structure

```text
├── app.py                            # Streamlit Web Application
├── damage_assessment_prototype.py    # Core pipeline & inference logic
├── requirements.txt                  # Python dependencies
├── models/                           # Store SAM checkpoints here
│   └── sam_vit_b_01ec64.pth          
├── weights/                          # Store trained classifier weights here
│   └── damage_classifier.pt          
├── data/                             # Put imagery pairs here
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── outputs/                          # Generated predictions & CSVs
```

## 🧠 Model Details
* **Segmentation Backbone**: Meta SAM (`vit_b`) loaded via `sam_model_registry`. Configured for robust building extraction (`pred_iou_thresh=0.88`, `stability_score_thresh=0.93`).
* **Damage Classifier**: Dual-branch `ResNet18` encoder. Feeds pre and post-disaster features into a fused Multi-Layer Perceptron (MLP) head to yield a 4-class softmax probability. *(Defaults to untrained ImageNet weights if `weights/damage_classifier.pt` is not present).*

## License
MIT License

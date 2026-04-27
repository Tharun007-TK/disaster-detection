# DAMAGESCOPE Backend — Render Deployment

Render free tier has **512 MB RAM** and **no GPU**. The settings below are
mandatory; deviating from any of them will reintroduce the OOM kill we just
fixed.

## Render service config

| Field | Value |
|---|---|
| Environment | Python 3.10+ |
| Root directory | repository root (so paths like `ml/checkpoints/best.pth` resolve) |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check path | `/docs` (FastAPI built-in) |

`--workers 1` is non-negotiable on the free tier: each worker is a full
process with its own torch + model copy. Two workers = 2x memory = OOM.

## Required environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_WEIGHTS_PATH` | `<repo>/ml/checkpoints/best.pth` | Path to trained Siamese checkpoint. |
| `OUTPUTS_DIR` | `./outputs` | Where inference results land. On Render this is ephemeral disk. |
| `EAGER_MODEL_LOAD` | unset | Set to `1` only in local dev to load the model at startup. **Do not set on Render.** |

## Memory budget after fixes

Rough per-component RSS on Render free tier:

| Component | RAM |
|---|---|
| Python + uvicorn + FastAPI baseline | ~80 MB |
| torch (CPU-only wheel) on import | ~150 MB |
| Siamese ResNet18 weights in RAM | ~50 MB |
| Per-request peak (input tensors + logits + numpy buffers) | ~80 MB |
| Headroom | ~150 MB |
| **Total peak** | **~360–410 MB** |

This sits comfortably under the 512 MB hard limit.

## Why the deploy was OOMing before

1. `requirements.txt` had bare `torch` / `torchvision` -> pip pulled the CUDA
   wheel (~800 MB on disk, ~600 MB resident after import). That alone
   exceeded the 512 MB cap.
2. Model loaded eagerly inside the FastAPI lifespan — the cold-start spike
   (uvicorn boot + torch init + checkpoint load) all hit at once.
3. Inference path used `@torch.no_grad()` instead of `inference_mode()` and
   never released the float32 logits tensor; peak request RAM was ~2x what
   it needed to be.
4. `_load_pair` made an explicit `.copy()` of the post image just to keep
   it for the overlay render, when a reference would have done.

## Fixes applied

| File | Fix | Approx. saving |
|---|---|---|
| `backend/requirements.txt` | Pin `torch==2.6.0+cpu`, `torchvision==0.21.0+cpu` via PyTorch CPU index. Drop training-only deps (albumentations, matplotlib, pandas, tqdm, scikit-learn). | **~450 MB on disk, ~400 MB RAM** |
| `backend/state.py` | Lazy, lock-guarded `get_model()`. `release_model()` for shutdown. | Defers ~200 MB until first request — prevents cold-boot spike |
| `backend/main.py` | Lifespan no longer eager-loads model. `EAGER_MODEL_LOAD=1` opt-in for local dev. | Cold-boot RAM stays ~80 MB until traffic |
| `backend/routes/inference.py` | Uses `state.get_model()`. `gc.collect()` in `finally` after each request. | Peak request RAM drops back to baseline between calls |
| `ml/inference.py` | `predict_mask` switched to `torch.inference_mode()`, drops `pre_p`/`post_p`/`logits` explicitly, casts to `uint8` before `.cpu()`. `_load_pair` / `_load_single` no longer `.copy()` original arrays for overlay; references suffice. | ~50–80 MB on peak per request |

## Behaviour caveats

- **First request after a cold start is slow** (~3–5 s) because the model
  loads on demand. Subsequent requests are hot. If you need a warm boot,
  set `EAGER_MODEL_LOAD=1` — but only do this on a paid tier with more RAM.
- All other request semantics, response shapes, and output filenames are
  unchanged. The frontend needs no updates.

## If memory is still tight

In rough order of effort vs. payoff:

1. **TorchScript the model on first load** — `torch.jit.script(model)`
   sometimes shrinks runtime memory and speeds inference; no behaviour
   change, no retrain.
2. **Stream large GeoTIFFs with `rasterio.windows.Window`** — read tiles
   instead of the whole array. Only matters for inputs >50 MP.
3. **ONNX Runtime CPU** — drop torch entirely at serve time. Big rewrite.
   Don't go here unless 1 and 2 aren't enough.

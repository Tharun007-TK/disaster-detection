"""DAMAGESCOPE FastAPI backend.

Run: uvicorn backend.main:app --reload --port 8000

Memory note (Render free tier, 512 MB):
The model is NOT loaded at startup. The first inference request triggers a
lazy load via `backend.state.get_model()`. This avoids the cold-boot RAM
spike (torch init + model weights + uvicorn worker baseline all at once)
that was tripping Render's OOM killer.
Set EAGER_MODEL_LOAD=1 in env to opt back into eager startup for local dev.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.state as state


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("EAGER_MODEL_LOAD") == "1":
        print("[lifespan] EAGER_MODEL_LOAD=1, warming model")
        state.get_model()
    else:
        print("[lifespan] lazy model load enabled (first request will warm)")
    yield
    state.release_model()


app = FastAPI(title="DAMAGESCOPE", version="2.0.0", lifespan=lifespan)

_cors_env = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_env:
    _allow_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    _allow_credentials = True
else:
    _allow_origins = ["*"]
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

from backend.routes import inference as _inf
from backend.routes import precautions as _prec
from backend.routes import results as _res

app.include_router(_inf.router, prefix="/api")
app.include_router(_res.router, prefix="/api")
app.include_router(_prec.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": state.is_loaded()}


@app.get("/")
def root():
    return {"service": "DAMAGESCOPE", "version": "2.0.0", "docs": "/docs"}

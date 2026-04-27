"""DAMAGESCOPE FastAPI backend.

Run: uvicorn backend.main:app --reload --port 8000
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
from ml.inference import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = Path(os.environ.get("MODEL_WEIGHTS_PATH", "D:\\Projects\\Disaster Damage Prediction\\Code\\v2\\ml\\checkpoints\\best.pth"))
    print(f"loading model from {ckpt}")
    state.model, state.device = load_model(ckpt)
    print(f"model ready on {state.device}")
    yield
    state.model = None
    state.device = None


app = FastAPI(title="DAMAGESCOPE", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

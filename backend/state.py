"""Lazy, thread-safe model holder.

The model is NOT loaded at startup on memory-constrained hosts (Render free
tier = 512 MB). Instead `get_model()` materialises it on the first inference
request and caches it for the process lifetime.

Trade-off: the first request after a cold start pays a ~3–5 s load penalty.
Every subsequent request is hot.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import torch

from ml.model import SiameseDamageNet

model: Optional[SiameseDamageNet] = None
device: Optional[torch.device] = None

_lock = threading.Lock()


def _resolve_ckpt_path() -> Path:
    return Path(
        os.environ.get(
            "MODEL_WEIGHTS_PATH",
            str(Path(__file__).resolve().parent.parent / "ml" / "checkpoints" / "best.pth"),
        )
    )


def get_model() -> tuple[SiameseDamageNet, torch.device]:
    """Return the cached model, loading it on first call.

    Thread-safe: concurrent requests during a cold start block on the same
    lock; only one actually runs `load_model`.
    """
    global model, device
    if model is not None and device is not None:
        return model, device
    with _lock:
        if model is None or device is None:
            # Local import keeps `ml.inference` (and its torch/rasterio deps)
            # out of the import graph until first use.
            from ml.inference import load_model

            ckpt = _resolve_ckpt_path()
            print(f"[state] lazy-loading model from {ckpt}")
            m, d = load_model(ckpt)
            m.eval()
            model = m
            device = d
            print(f"[state] model ready on {device}")
    return model, device  # type: ignore[return-value]


def release_model() -> None:
    """Drop the cached model. Call from app shutdown."""
    global model, device
    with _lock:
        model = None
        device = None

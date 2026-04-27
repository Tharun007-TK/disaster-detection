from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import backend.state as state
from ml.inference import run_inference, run_single_inference

router = APIRouter()

_ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _validate(upload: UploadFile, label: str) -> None:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, f"{label} must be .tif/.tiff, got {ext!r}")


@router.post("/inference")
async def do_inference(
    pre: UploadFile = File(..., description="Pre-event GeoTIFF (3-band RGB)"),
    post: UploadFile = File(..., description="Post-event GeoTIFF (1-band SAR)"),
):
    _validate(pre, "pre")
    _validate(post, "post")

    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    out_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))

    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path = Path(tmpdir) / (pre.filename or "pre.tif")
        post_path = Path(tmpdir) / (post.filename or "post.tif")
        with pre_path.open("wb") as f:
            shutil.copyfileobj(pre.file, f)
        with post_path.open("wb") as f:
            shutil.copyfileobj(post.file, f)

        try:
            result = run_inference(
                pre_path,
                post_path,
                ckpt_path=Path(os.environ.get("MODEL_WEIGHTS_PATH", "./ml/checkpoints/best.pth")),
                out_dir=out_dir,
                model=state.model,
                device=state.device,
            )
        except Exception as exc:
            raise HTTPException(500, f"Inference failed: {exc}") from exc

    return result


@router.post("/inference/single")
async def do_single_inference(
    image: UploadFile = File(..., description="Single disaster GeoTIFF (any band count)"),
):
    _validate(image, "image")
    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    out_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / (image.filename or "image.tif")
        with img_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        try:
            result = run_single_inference(
                img_path,
                ckpt_path=Path(os.environ.get("MODEL_WEIGHTS_PATH", "./ml/checkpoints/best.pth")),
                out_dir=out_dir,
                model=state.model,
                device=state.device,
            )
        except Exception as exc:
            raise HTTPException(500, f"Inference failed: {exc}") from exc

    return result

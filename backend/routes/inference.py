from __future__ import annotations

import asyncio
import gc
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import backend.state as state

log = logging.getLogger("damagescope.inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

router = APIRouter()

_ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _validate(upload: UploadFile, label: str) -> None:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, f"{label} must be .tif/.tiff, got {ext!r}")


def _ckpt_path() -> Path:
    return Path(
        os.environ.get(
            "MODEL_WEIGHTS_PATH",
            str(Path(__file__).resolve().parent.parent.parent / "ml" / "checkpoints" / "best.pth"),
        )
    )


@router.post("/inference")
async def do_inference(
    pre: UploadFile = File(..., description="Pre-event GeoTIFF (3-band RGB)"),
    post: UploadFile = File(..., description="Post-event GeoTIFF (1-band SAR)"),
):
    _validate(pre, "pre")
    _validate(post, "post")

    # Lazy import keeps ml.inference (and its torch + rasterio import cost)
    # out of FastAPI startup graph; only paid on first inference call.
    from ml.inference import run_inference

    try:
        model, device = state.get_model()
    except Exception as exc:
        raise HTTPException(503, f"Model load failed: {exc}") from exc

    out_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))

    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path = Path(tmpdir) / (pre.filename or "pre.tif")
        post_path = Path(tmpdir) / (post.filename or "post.tif")
        with pre_path.open("wb") as f:
            shutil.copyfileobj(pre.file, f)
        with post_path.open("wb") as f:
            shutil.copyfileobj(post.file, f)

        log.info("pair inference start: pre=%s (%d bytes) post=%s (%d bytes)",
                 pre.filename, pre_path.stat().st_size,
                 post.filename, post_path.stat().st_size)
        try:
            result = await asyncio.to_thread(
                run_inference,
                pre_path,
                post_path,
                ckpt_path=_ckpt_path(),
                out_dir=out_dir,
                model=model,
                device=device,
            )
        except Exception as exc:
            log.error("pair inference failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(500, f"Inference failed: {exc.__class__.__name__}: {exc}") from exc
        finally:
            gc.collect()

    log.info("pair inference done: %s", result.get("id") if isinstance(result, dict) else "ok")
    return result


@router.post("/inference/single")
async def do_single_inference(
    image: UploadFile = File(..., description="Single disaster GeoTIFF (any band count)"),
):
    _validate(image, "image")

    from ml.inference import run_single_inference

    try:
        model, device = state.get_model()
    except Exception as exc:
        raise HTTPException(503, f"Model load failed: {exc}") from exc

    out_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / (image.filename or "image.tif")
        with img_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        log.info("single inference start: image=%s (%d bytes)",
                 image.filename, img_path.stat().st_size)
        try:
            result = await asyncio.to_thread(
                run_single_inference,
                img_path,
                ckpt_path=_ckpt_path(),
                out_dir=out_dir,
                model=model,
                device=device,
            )
        except Exception as exc:
            log.error("single inference failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(500, f"Inference failed: {exc.__class__.__name__}: {exc}") from exc
        finally:
            gc.collect()

    log.info("single inference done: %s", result.get("id") if isinstance(result, dict) else "ok")
    return result

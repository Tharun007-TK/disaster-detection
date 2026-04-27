from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from backend.utils import list_results

router = APIRouter()


@router.get("/results")
def get_results():
    out_dir = Path(os.environ.get("OUTPUTS_DIR", "./outputs"))
    return list_results(out_dir)

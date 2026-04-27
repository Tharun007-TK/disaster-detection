from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _bounds_from_tif(tif_path: Path) -> tuple[dict | None, str | None]:
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            b = src.bounds
            crs = str(src.crs) if src.crs else None
            return {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top}, crs
    except Exception:
        return None, None


def _to_wgs84_center(bounds: dict | None, crs_str: str | None) -> tuple[float | None, float | None]:
    """Reproject any CRS to WGS84 center. Returns (None, None) if no valid CRS or out of range."""
    if not bounds or not crs_str:
        return None, None
    try:
        import rasterio.warp
        from rasterio.crs import CRS
        src_crs = CRS.from_string(crs_str)
        left, bottom, right, top = rasterio.warp.transform_bounds(
            src_crs, CRS.from_epsg(4326),
            bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
        )
        lat = (bottom + top) / 2
        lng = (left + right) / 2
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    except Exception:
        pass
    return None, None


def list_results(outputs_dir: Path) -> list[dict]:
    results = []
    for json_file in sorted(
        outputs_dir.glob("*_summary.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(json_file.read_text())
            stem = json_file.stem.replace("_summary", "")
            mask_png = json_file.parent / f"{stem}_mask.png"
            mask_tif = json_file.parent / f"{stem}_mask.tif"
            overlay_png = json_file.parent / f"{stem}_overlay.png"
            if not overlay_png.exists():
                overlay_png = json_file.parent / f"{stem}_single_overlay.png"
            bounds = data.get("bounds")
            crs_str = data.get("crs")
            if bounds is None:
                bounds, crs_str = _bounds_from_tif(mask_tif)
            lat, lng = _to_wgs84_center(bounds, crs_str)
            results.append(
                {
                    "id": stem,
                    "stem": stem,
                    "damage_pct": data.get("damage_pct", {}),
                    "mask_url": f"/outputs/{mask_png.name}" if mask_png.exists() else None,
                    "overlay_url": f"/outputs/{overlay_png.name}" if overlay_png.exists() else None,
                    "timestamp": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                    "bounds": bounds,
                    "lat": lat,
                    "lng": lng,
                    **{
                        k: data[k]
                        for k in ("pixel_counts", "height", "width", "crs", "num_classes")
                        if k in data
                    },
                }
            )
        except Exception:
            continue
    return results

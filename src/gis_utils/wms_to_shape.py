#!/usr/bin/env python3
"""
Download a WMS layer for an extent, detect green boundary lines, vectorize to parcel polygons, optionally
fill interior rings and query GetFeatureInfo, export shapefile. Generic tool for any WMS with line-style layers.

CLI: conda activate gis && python -m gis_utils.wms_to_shape --input-boundary input/area.shp --output Shapes/out.shp [options]

Library (no defaults — you must specify extent or input_boundary, and one of line_color, line_color_rgb, or line_channel):
  from pathlib import Path
  from gis_utils.wms_to_shape import run, get_bounds_from_shape

  # Extent from a boundary shapefile (optional buffer to expand area):
  bounds = get_bounds_from_shape(Path("input/area.shp"), crs="EPSG:25833", buffer_m=10.0)
  gdf = run(bounds, Path("Shapes/out.shp"), wms_url="https://...", wms_layer="LayerName", line_color="green")

  # Or pass input_boundary and let run() compute extent (with optional boundary_buffer_m):
  gdf = run(None, Path("Shapes/out.shp"), input_boundary=Path("input/area.shp"), boundary_buffer_m=10.0,
            wms_url="...", wms_layer="...", line_color="green")

  # map_size=None auto-detects max resolution from WMS GetCapabilities (MaxWidth/MaxHeight).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force unbuffered stdout so logs appear immediately (e.g. under conda run)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("(importing geopandas...)", flush=True)
import geopandas as gpd
print("(importing PIL, numpy, rasterio...)", flush=True)
from PIL import Image
import numpy as np
import rasterio
import requests
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union, polygonize, split
print("(imports done)", flush=True)

# -----------------------------------------------------------------------------
# Config (defaults; overridable via run() or CLI)
# -----------------------------------------------------------------------------
DOWNLOAD_ONLY = False
SKIP_GETFEATUREINFO = True

# WMS: None for map_size = auto-detect from GetCapabilities (MaxWidth/MaxHeight)
DEFAULT_WMS_URL = "https://www.geodaten-mv.de/dienste/bodenschaetzwerte_wms"
DEFAULT_WMS_LAYER = "BOSIS_MV_gesamt"
DEFAULT_CRS = "EPSG:25833"
DEFAULT_MAP_SIZE: int | None = None  # None = auto (highest detail from GetCapabilities)
CACHE_DIR_NAME = "wms_cache"
REQUEST_DELAY_S = 0.2
GFI_WORKERS = 8  # GetFeatureInfo parallel requests; 1 = sequential (with REQUEST_DELAY_S)
MIN_PARCEL_AREA_M2 = 500.0
SIMPLIFY_TOLERANCE_M = 1.0
USE_GRASS = True
SNAP_GRID_M = 0.01
SIMPLIFY_BLADE_M = 1.0
MAX_HOLE_AREA_SQ_M = None

# Line detection: which pixels are "boundary lines" (default: green preset for Bodenschätzung-style WMS)
DEFAULT_LINE_COLOR = "green"  # preset: green|black|red|blue|white
DEFAULT_LINE_COLOR_RGB: tuple[int, int, int, int, int, int] | None = None  # custom: (r_min,r_max,g_min,g_max,b_min,b_max)
DEFAULT_BACKGROUND_COLOR: tuple[int, int, int] | None = None  # (r,g,b); if set, line = pixel not matching background
DEFAULT_BACKGROUND_TOLERANCE = 15  # max distance from background to count as background
DEFAULT_LINE_CHANNEL: str | None = None  # "alpha"|"0"|"1"|"2" = use single channel; None = use color
DEFAULT_LINE_CHANNEL_MIN = 128  # for alpha: line where alpha >= min
DEFAULT_LINE_CHANNEL_MAX = 255  # for alpha; for RGB dark line use 0,80

SCRIPT_DIR = Path(__file__).resolve().parent
# When used as a library, defaults are cwd-based; pass explicit paths in run().
PROJECT_ROOT = Path.cwd()
DEFAULT_INPUT_BOUNDARY = None  # set per project or pass input_boundary= to run()
DEFAULT_OUTPUT_SHP = None  # pass output_path= to run()
CACHE_DIR = PROJECT_ROOT / CACHE_DIR_NAME

# Backwards compatibility
CRS = DEFAULT_CRS
WMS_URL = DEFAULT_WMS_URL
WMS_LAYER = DEFAULT_WMS_LAYER
MAP_SIZE = DEFAULT_MAP_SIZE or 4192


def get_wms_max_size(wms_url: str, timeout: int = 30) -> tuple[int, int] | None:
    """Query WMS GetCapabilities and return (MaxWidth, MaxHeight) if present; else None."""
    params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetCapabilities"}
    try:
        r = requests.get(wms_url, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        mw = mh = None
        for e in root.iter():
            tag = e.tag.split("}")[-1] if "}" in e.tag else e.tag
            if tag == "MaxWidth" and e.text:
                mw = int(e.text.strip())
            elif tag == "MaxHeight" and e.text:
                mh = int(e.text.strip())
        if mw is not None and mh is not None and mw > 0 and mh > 0:
            return (mw, mh)
    except Exception:
        pass
    return None


def get_bounds_from_shape(
    shape_path: Path | str,
    crs: str = DEFAULT_CRS,
    buffer_m: float = 10.0,
) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) from the bounding box of the shapefile, in given CRS, optionally buffered."""
    shape_path = Path(shape_path)
    gdf = gpd.read_file(shape_path).to_crs(crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    if buffer_m > 0:
        w, h = maxx - minx, maxy - miny
        margin = buffer_m if (w == 0 and h == 0) else max(buffer_m, max(w, h) * 0.02)
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
    return float(minx), float(miny), float(maxx), float(maxy)


def _build_line_mask(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    *,
    line_color: str | None = None,
    line_color_rgb: tuple[int, int, int, int, int, int] | None = None,
    background_color: tuple[int, int, int] | None = None,
    background_tolerance: int = 15,
    line_channel: str | None = None,
    line_channel_min: int = 128,
    line_channel_max: int = 255,
) -> np.ndarray:
    """
    Build binary mask of "line" pixels from RGB (and optional alpha).
    Returns uint8 array (1 = line, 0 = not line).

    - line_channel set: use single channel (alpha or R/G/B band); line where value in [min, max].
    - Else: use line_color preset or line_color_rgb (r_min,r_max,g_min,g_max,b_min,b_max).
    - If background_color set: exclude pixels within tolerance of that color (line = not background).
    """
    R = rgb[..., 0].astype(np.int32)
    G = rgb[..., 1].astype(np.int32)
    B = rgb[..., 2].astype(np.int32)

    if line_channel is not None:
        ch = line_channel.lower()
        if ch == "alpha":
            if alpha is None:
                # No alpha: treat as all opaque (all line) or use luminance; treat as no line
                line_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            else:
                a = alpha.astype(np.int32)
                line_mask = ((a >= line_channel_min) & (a <= line_channel_max)).astype(np.uint8)
        elif ch in ("0", "1", "2"):
            band = (R, G, B)[int(ch)]
            line_mask = ((band >= line_channel_min) & (band <= line_channel_max)).astype(np.uint8)
        else:
            line_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    else:
        if line_color_rgb is not None:
            r_lo, r_hi, g_lo, g_hi, b_lo, b_hi = line_color_rgb
            line_mask = (
                (R >= r_lo) & (R <= r_hi) &
                (G >= g_lo) & (G <= g_hi) &
                (B >= b_lo) & (B <= b_hi)
            ).astype(np.uint8)
        else:
            if line_color is None:
                raise ValueError("line_color is required when not using line_color_rgb or line_channel")
            preset = line_color.lower()
            if preset == "green":
                line_mask = ((G > R) & (G > B) & (G > 60)).astype(np.uint8)
            elif preset == "black":
                line_mask = ((R < 80) & (G < 80) & (B < 80)).astype(np.uint8)
            elif preset == "red":
                line_mask = ((R > G) & (R > B) & (R > 60)).astype(np.uint8)
            elif preset == "blue":
                line_mask = ((B > R) & (B > G) & (B > 60)).astype(np.uint8)
            elif preset == "white":
                line_mask = ((R > 200) & (G > 200) & (B > 200)).astype(np.uint8)
            else:
                line_mask = ((G > R) & (G > B) & (G > 60)).astype(np.uint8)

    if background_color is not None and background_tolerance >= 0:
        r0, g0, b0 = background_color
        dist = np.maximum(np.abs(R - r0), np.maximum(np.abs(G - g0), np.abs(B - b0)))
        not_background = (dist > background_tolerance).astype(np.uint8)
        line_mask = (line_mask & not_background).astype(np.uint8)

    return line_mask


def download_wms_raster(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    wms_url: str | None = None,
    wms_layer: str | None = None,
    crs: str | None = None,
    map_size: int | None = None,
    cache_dir: Path | None = None,
    cache_key_prefix: str = "wms",
) -> Path:
    """Download WMS image for bbox, save as georeferenced GeoTIFF in cache. Returns path.
    map_size: pixels on longest side; None = auto from GetCapabilities (highest detail).
    """
    wms_url = wms_url or DEFAULT_WMS_URL
    wms_layer = wms_layer or DEFAULT_WMS_LAYER
    crs = crs or DEFAULT_CRS
    cache_dir = cache_dir or CACHE_DIR
    if map_size is None:
        cap = get_wms_max_size(wms_url)
        map_size = min(cap) if cap else 4096
        print(f"[download] Map size from GetCapabilities: {map_size} px (longest side)", flush=True)
    else:
        map_size = int(map_size)

    print("[download] Ensuring cache dir exists...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    w_deg = maxx - minx
    h_deg = maxy - miny
    longest = max(w_deg, h_deg)
    if longest <= 0:
        width = height = map_size
    else:
        width = max(64, int(map_size * w_deg / longest))
        height = max(64, int(map_size * h_deg / longest))
    print(f"[download] Request size: {width} x {height} px for bbox ({minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f})")

    cache_name = f"{cache_key_prefix}_{minx:.0f}_{miny:.0f}_{maxx:.0f}_{maxy:.0f}_{width}x{height}.tif"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"[download] Using cached file: {cache_path} ({size_mb:.2f} MB)")
        try:
            with rasterio.open(cache_path) as src:
                print(f"[download] Cache OK: readable, {src.width}x{src.height} px, {src.count} bands.")
        except Exception as e:
            print(f"[download] WARNING: cached file exists but could not be opened: {e}")
        return cache_path

    print(f"[download] Requesting GetMap from WMS (this may take a while for large areas)...")
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": wms_layer,
        "STYLES": "default",
        "CRS": crs,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
    }
    r = requests.get(wms_url, params=params, timeout=120)
    size_bytes = len(r.content)
    size_mb = size_bytes / 1024 / 1024
    content_length = r.headers.get("Content-Length")
    print(f"[download] Response: status={r.status_code}, Content-Type={r.headers.get('Content-Type', '')}, size={size_mb:.2f} MB")
    if content_length is not None:
        expected = int(content_length)
        if size_bytes == expected:
            print(f"[download] Content-Length matches received bytes ({expected}) — full download verified.")
        else:
            print(f"[download] WARNING: received {size_bytes} bytes but Content-Length was {expected} — download may be incomplete.")
    else:
        print("[download] No Content-Length header; cannot verify full download (rely on image parsing).")
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if "xml" in ct or r.content.lstrip().startswith(b"<?xml"):
        raise RuntimeError(f"WMS returned error instead of image: {r.text[:500]}")
    print("[download] Parsing image...")
    arr = np.array(Image.open(io.BytesIO(r.content)))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    print("[download] Writing georeferenced GeoTIFF to cache...")
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    with rasterio.open(
        cache_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr[:, :, 0], 1)
        dst.write(arr[:, :, 1], 2)
        dst.write(arr[:, :, 2], 3)
    print(f"[download] Done. Cached at: {cache_path}")
    return cache_path


def _log_elapsed(msg: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  [segment] {msg} ({elapsed:.1f}s)", flush=True)
    return time.perf_counter()


def _thin_polygon_centerline(poly, grid_size: int = 80) -> LineString | None:
    """Get centerline of a thin polygon (so boundary becomes a shared edge). Returns LineString or None."""
    from rasterio.features import rasterize
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None
    minx, miny, maxx, maxy = poly.bounds
    w = h = grid_size
    transform = from_bounds(minx, miny, maxx, maxy, w, h)
    raster = rasterize([(poly, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
    if raster.sum() < 4:
        return None
    skel = skeletonize(raster.astype(bool)).astype(np.uint8)
    if skel.sum() < 2:
        return None
    # Trace skeleton to one LineString (take longest path)
    from scipy.ndimage import label
    labeled, n_comp = label(skel, structure=np.ones((3, 3)))
    best_line = None
    best_len = 0
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(labeled == comp_id)
        pixels = list(zip(ys.tolist(), xs.tolist()))
        if len(pixels) < 2:
            continue
        coords = [transform * (c, r) for r, c in pixels]
        line = LineString(coords)
        if line.length > best_len:
            best_len = line.length
            best_line = line
    return best_line


def _centerlines_via_grass(line_mask: np.ndarray, transform, crs) -> list | None:
    """
    Use GRASS r.thin + r.to.vect to get centerlines from line mask. Returns list of LineStrings or None on failure.
    """
    import subprocess
    import tempfile
    # Write line mask as GeoTIFF (1=line, 0=background; GRASS will set 0 to null)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        green_tif = Path(f.name)
    try:
        h, w = line_mask.shape
        # Int32, 1 where line, 0 else; nodata=0 for r.thin
        arr = np.where(line_mask.astype(bool), 1, 0).astype(np.int32)
        with rasterio.open(
            green_tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(arr, 1)
    except Exception as e:
        print(f"  [segment] GRASS: failed to write green mask tif: {e}", flush=True)
        green_tif.unlink(missing_ok=True)
        return None

    grass_db = Path(tempfile.mkdtemp(prefix="grassdb_wms_"))
    out_geojson = grass_db / "centerlines.geojson"
    try:
        # Create GRASS location from the georeferenced raster
        create_cmd = [
            "grass",
            "-c", str(green_tif),
            str(grass_db / "loc"),
            "-e",
        ]
        print(f"  [segment] GRASS: creating location and running r.thin + r.to.vect...", flush=True)
        r = subprocess.run(create_cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  [segment] GRASS: grass -c failed: {r.stderr[:500]}", flush=True)
            return None
        # Run centerline extraction inside GRASS
        exec_cmd = [
            "grass",
            str(grass_db / "loc" / "PERMANENT"),
            "--exec",
            "python",
            str(SCRIPT_DIR / "grass_centerlines.py"),
            str(green_tif),
            str(out_geojson),
        ]
        r = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  [segment] GRASS: worker failed: {r.stderr[:500] if r.stderr else r.stdout[:500]}", flush=True)
            return None
        if not out_geojson.exists():
            print(f"  [segment] GRASS: output GeoJSON not created", flush=True)
            return None
        # Read centerlines from GeoJSON
        gdf = gpd.read_file(out_geojson)
        if gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
        else:
            gdf = gdf.to_crs(crs)
        line_segments = []
        for geom in gdf.geometry:
            if geom.is_empty:
                continue
            if geom.geom_type == "LineString" and len(geom.coords) >= 2:
                line_segments.append(LineString(geom.coords))
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    if len(part.coords) >= 2:
                        line_segments.append(LineString(part.coords))
        return line_segments
    except subprocess.TimeoutExpired:
        print(f"  [segment] GRASS: timed out", flush=True)
        return None
    except Exception as e:
        print(f"  [segment] GRASS: {e}", flush=True)
        return None
    finally:
        green_tif.unlink(missing_ok=True)
        try:
            import shutil
            shutil.rmtree(grass_db, ignore_errors=True)
        except Exception:
            pass


def _snap_line_to_grid(line, grid_size: float) -> LineString:
    """Snap all coordinates to grid so shared vertices coincide (reduces slivers)."""
    if line is None or line.is_empty or len(line.coords) < 2:
        return line
    coords = []
    for x, y in line.coords:
        x = round(x / grid_size) * grid_size
        y = round(y / grid_size) * grid_size
        coords.append((x, y))
    return LineString(coords)


def _extent_box_segments(minx: float, miny: float, maxx: float, maxy: float) -> list:
    """Return the 4 edges of the extent box as LineStrings (closed ring)."""
    return [
        LineString([(minx, miny), (maxx, miny)]),
        LineString([(maxx, miny), (maxx, maxy)]),
        LineString([(maxx, maxy), (minx, maxy)]),
        LineString([(minx, maxy), (minx, miny)]),
    ]


def _nearest_point_on_box(x: float, y: float, minx: float, miny: float, maxx: float, maxy: float) -> tuple[float, float]:
    """Return the nearest point on the box boundary to (x,y)."""
    if minx <= x <= maxx and miny <= y <= maxy:
        # Inside: project to nearest side
        dl, dr = x - minx, maxx - x
        db, dt = y - miny, maxy - y
        d = min(dl, dr, db, dt)
        if d == dl:
            return (minx, y)
        if d == dr:
            return (maxx, y)
        if d == db:
            return (x, miny)
        return (x, maxy)
    # Outside: clamp to box
    x2 = max(minx, min(maxx, x))
    y2 = max(miny, min(maxy, y))
    return (x2, y2)


def _extract_lines_from_polygon(poly) -> list:
    """Return list of LineStrings from a polygon's exterior and interiors."""
    lines = []
    if hasattr(poly, "exterior") and poly.exterior is not None and len(poly.exterior.coords) >= 2:
        lines.append(LineString(poly.exterior.coords))
    for interior in getattr(poly, "interiors", []) or []:
        if interior is not None and len(interior.coords) >= 2:
            lines.append(LineString(interior.coords))
    return lines


def _skeleton_mask_to_lines(mask: np.ndarray, transform) -> list:
    """
    Skeletonize the green mask (full raster resolution) and trace each skeleton component
    to ordered LineStrings in world coords. Returns 1D centerlines so split() won't create thin polygons.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return []
    from scipy.ndimage import label
    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    if skel.sum() < 2:
        return []
    h, w = skel.shape
    # 8-neighbour connectivity
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = label(skel, structure=structure)
    out_lines = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(labeled == comp_id)
        pixels = set(zip(ys.tolist(), xs.tolist()))
        if len(pixels) < 2:
            continue
        # Neighbours (8-connected)
        def neighbours(r, c):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pixels:
                        yield (nr, nc)
        # Endpoints: exactly one neighbour
        endpoints = [p for p in pixels if sum(1 for _ in neighbours(*p)) == 1]
        # Trace each branch (from an endpoint to next endpoint or junction)
        used = set()
        for start in endpoints:
            if start in used:
                continue
            path = [start]
            used.add(start)
            current = start
            while True:
                next_candidates = [n for n in neighbours(*current) if n not in used]
                if not next_candidates:
                    break
                next_p = next_candidates[0]
                if len(next_candidates) > 1 and len(path) >= 2:
                    pr, pc = path[-2]
                    dr, dc = current[0] - pr, current[1] - pc
                    for n in next_candidates:
                        if (n[0] - current[0]) == dr and (n[1] - current[1]) == dc:
                            next_p = n
                            break
                path.append(next_p)
                used.add(next_p)
                current = next_p
                if sum(1 for _ in neighbours(*current)) != 2 and len(path) >= 2:
                    break  # hit endpoint or junction
            if len(path) >= 2:
                coords = [transform * (c, r) for r, c in path]
                out_lines.append(LineString(coords))
        # If no endpoints (loop), take longest path through the component
        if not endpoints and len(pixels) >= 2:
            start = next(iter(pixels))
            path = [start]
            current = start
            used = {start}
            while True:
                next_candidates = [n for n in neighbours(*current) if n not in used]
                if not next_candidates:
                    break
                next_p = next_candidates[0]
                path.append(next_p)
                used.add(next_p)
                current = next_p
            if len(path) >= 2:
                coords = [transform * (c, r) for r, c in path]
                out_lines.append(LineString(coords))
    return out_lines


def segment_and_vectorize(
    raster_path: Path,
    min_area_m2: float | None = None,
    *,
    line_color: str | None = None,
    line_color_rgb: tuple[int, int, int, int, int, int] | None = None,
    background_color: tuple[int, int, int] | None = None,
    background_tolerance: int | None = None,
    line_channel: str | None = None,
    line_channel_min: int | None = None,
    line_channel_max: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Boundary-based: detect line pixels (by color preset, custom RGB, or single channel),
    vectorize to line segments, polygonize to parcel polygons. Thin polygons are dropped by min_area_m2.
    """
    if min_area_m2 is None:
        min_area_m2 = MIN_PARCEL_AREA_M2
    from scipy.ndimage import binary_dilation

    if line_color is None and line_color_rgb is None and line_channel is None:
        raise ValueError("One of line_color, line_color_rgb, or line_channel must be provided")
    _line_color = line_color
    _line_color_rgb = line_color_rgb
    _bg = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
    _bg_tol = background_tolerance if background_tolerance is not None else DEFAULT_BACKGROUND_TOLERANCE
    _ch = line_channel if line_channel is not None else DEFAULT_LINE_CHANNEL
    _ch_min = line_channel_min if line_channel_min is not None else DEFAULT_LINE_CHANNEL_MIN
    _ch_max = line_channel_max if line_channel_max is not None else DEFAULT_LINE_CHANNEL_MAX

    t = time.perf_counter()
    with rasterio.open(raster_path) as src:
        arr = src.read()  # (C, H, W)
        transform = src.transform
        crs = src.crs
    t = _log_elapsed("Read raster", t)

    rgb = np.transpose(arr[:3], (1, 2, 0))
    alpha = arr[3] if arr.shape[0] >= 4 else None  # (H, W)

    line_mask = _build_line_mask(
        rgb,
        alpha,
        line_color=_line_color,
        line_color_rgb=_line_color_rgb,
        background_color=_bg,
        background_tolerance=_bg_tol,
        line_channel=_ch,
        line_channel_min=_ch_min,
        line_channel_max=_ch_max,
    )
    t = _log_elapsed("Line mask", t)

    # Dilate slightly to connect dashed segments
    struct = np.ones((2, 2), dtype=np.uint8)
    line_mask = binary_dilation(line_mask, structure=struct).astype(np.uint8)
    t = _log_elapsed("Dilate (connect dashed)", t)

    h, w = line_mask.shape

    # Strategy: centerlines (1D) as the only linework -> polygonize -> each face is one polygon, edges shared (no thin polygons).
    from shapely.geometry import box
    minx, miny = transform * (0, h)
    maxx, maxy = transform * (w, 0)
    clip_box = box(minx, miny, maxx, maxy)
    tol = 0.5  # metres: treat as on box if within this

    if USE_GRASS:
        line_segments = _centerlines_via_grass(line_mask, transform, crs)
        if line_segments is not None:
            t = _log_elapsed(f"GRASS r.thin -> {len(line_segments)} centerline segments", t)
        else:
            line_segments = []
    else:
        line_segments = []
    if not line_segments:
        print(f"  [segment] Skeletonizing line mask -> centerlines (fallback)...", flush=True)
        line_segments = _skeleton_mask_to_lines(line_mask, transform)
        t = _log_elapsed(f"Skeleton -> {len(line_segments)} centerline segments", t)

    if not line_segments:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # Add extent box boundary (4 segments) so outer region is closed
    all_segments = list(line_segments) + _extent_box_segments(minx, miny, maxx, maxy)

    # Only extend endpoints that are near the box (within 2 m) — not every endpoint, to avoid
    # spurious lines cutting across parcels (which made the result "blocky" and misaligned)
    EXTEND_TO_BOX_DISTANCE_M = 2.0
    def on_box(x, y):
        return (abs(x - minx) <= tol or abs(x - maxx) <= tol or abs(y - miny) <= tol or abs(y - maxy) <= tol)
    def dist_to_box(x, y):
        if not (minx <= x <= maxx and miny <= y <= maxy):
            return float("inf")
        return min(x - minx, maxx - x, y - miny, maxy - y)
    extensions = []
    for line in line_segments:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        for pt in (coords[0], coords[-1]):
            x, y = pt[0], pt[1]
            if minx <= x <= maxx and miny <= y <= maxy and not on_box(x, y) and dist_to_box(x, y) <= EXTEND_TO_BOX_DISTANCE_M:
                nx, ny = _nearest_point_on_box(x, y, minx, miny, maxx, maxy)
                if (x, y) != (nx, ny):
                    extensions.append(LineString([(x, y), (nx, ny)]))
    all_segments.extend(extensions)
    if extensions:
        print(f"  [segment] Added {len(extensions)} extension segments (endpoints within {EXTEND_TO_BOX_DISTANCE_M} m of box)", flush=True)

    # Node so segments connect, then SPLIT extent by linework (not polygonize) -> full partition, no gaps
    t_node = time.perf_counter()
    print(f"  [segment] Noding and splitting extent by centerlines (no gaps)...", flush=True)
    noded = unary_union(all_segments)
    if noded.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    if noded.geom_type == "LineString":
        noded_lines = [noded]
    elif noded.geom_type == "MultiLineString":
        noded_lines = list(noded.geoms)
    else:
        noded_lines = [g for g in (noded.geoms if hasattr(noded, "geoms") else [noded]) if g.geom_type in ("LineString", "LinearRing")]
    t = _log_elapsed(f"Noded: {len(noded_lines)} segments", t_node)

    # Snap to grid so shared vertices coincide (fewer slivers); then simplify blade once so both sides
    # of each edge are identical (no overlaps) — we do NOT simplify polygons after.
    blade_lines = []
    for line in noded_lines:
        if line.is_empty or len(line.coords) < 2:
            continue
        line = _snap_line_to_grid(line, SNAP_GRID_M)
        if SIMPLIFY_BLADE_M > 0:
            line = line.simplify(SIMPLIFY_BLADE_M, preserve_topology=True)
        if not line.is_empty and len(line.coords) >= 2:
            blade_lines.append(line)
    if not blade_lines:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    blade = MultiLineString(blade_lines) if len(blade_lines) > 1 else blade_lines[0]
    try:
        split_result = split(clip_box, blade)
    except Exception as e:
        print(f"  [segment] Split failed ({e}), falling back to polygonize", flush=True)
        split_result = None
    regions = []
    if split_result is not None and hasattr(split_result, "geoms"):
        for poly in split_result.geoms:
            if poly.is_empty or not poly.is_valid or poly.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if poly.geom_type == "Polygon" and poly.area >= 1.0:
                regions.append(poly)
            elif poly.geom_type == "MultiPolygon":
                for part in poly.geoms:
                    if part.geom_type == "Polygon" and part.area >= 1.0:
                        regions.append(part)
    if not regions:
        # Fallback: polygonize (may leave gaps)
        for poly in polygonize(noded_lines):
            if poly.is_empty or not poly.is_valid or poly.area < 1.0:
                continue
            inter = poly.intersection(clip_box)
            if inter.is_empty:
                continue
            if inter.geom_type == "Polygon" and inter.area >= 1.0:
                regions.append(inter)
            elif inter.geom_type == "MultiPolygon":
                for part in inter.geoms:
                    if part.geom_type == "Polygon" and part.area >= 1.0:
                        regions.append(part)
    n_parcels = sum(1 for r in regions if r.area >= min_area_m2)
    if regions:
        areas = sorted(r.area for r in regions)
        print(f"  [segment] Split extent: {len(regions)} regions (m²): min={min(areas):.1f} max={max(areas):.1f} ({n_parcels} >= {min_area_m2})", flush=True)
    t = _log_elapsed(f"Split: {len(regions)} regions", t_node)

    if not regions:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # Return only parcels (area >= min); boundaries are already shared (no thin polygons)
    parcels = [r for r in regions if r.area >= min_area_m2]
    return gpd.GeoDataFrame(geometry=parcels, crs=crs)


def _boundaries_as_shared_edges(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Replace thin 'line' polygons with shared edges: for each thin polygon, find the two
    parcels it separates, get centerline of thin, split the union by centerline so the
    two parcels share that edge. Returns GeoDataFrame with only parcels (no thin polygons).
    """
    parcels_list = list(gdf[gdf["is_parcel"]].geometry)
    thin_list = list(gdf[~gdf["is_parcel"]].geometry)
    if not thin_list:
        return gdf[gdf["is_parcel"]].drop(columns=["is_parcel"], errors="ignore")

    # Shared-edge step runs before simplify so boundaries are exact; tiny buffer for float tolerance
    buf = 0.01  # metres
    n_replaced = 0
    for thin in thin_list:
        if thin.is_empty or thin.area < 1:
            continue
        thin_buf = thin.buffer(buf) if buf else thin
        touching = [
            i for i, p in enumerate(parcels_list)
            if p.buffer(buf).intersects(thin_buf) and not p.equals(thin)
        ]
        if len(touching) < 2:
            continue
        # If more than 2 (e.g. T-junction), take the 2 with largest intersection with thin
        if len(touching) > 2:
            touching = sorted(
                touching,
                key=lambda i: parcels_list[i].intersection(thin).area,
                reverse=True,
            )[:2]
        A, B = parcels_list[touching[0]], parcels_list[touching[1]]
        centerline = _thin_polygon_centerline(thin)
        if centerline is None or centerline.is_empty or centerline.length < 0.1:
            # Fallback: use thin polygon's boundary (outline) as splitter
            if thin.boundary and not thin.boundary.is_empty and thin.boundary.length >= 0.1:
                centerline = thin.boundary if thin.boundary.geom_type == "LineString" else LineString(thin.boundary.coords)
            else:
                continue
        if centerline is None or centerline.is_empty or centerline.length < 0.1:
            continue
        try:
            combined = unary_union([A, thin, B])
            if combined.is_empty or combined.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            split_result = split(combined, centerline)
            if not hasattr(split_result, "geoms"):
                continue
            parts = list(split_result.geoms)
            parts = [p for p in parts if p.geom_type == "Polygon" and not p.is_empty and p.area >= MIN_PARCEL_AREA_M2]
            if len(parts) != 2:
                continue
            for idx in sorted(touching, reverse=True):
                parcels_list.pop(idx)
            parcels_list.extend(parts)
            n_replaced += 1
        except Exception:
            continue

    print(f"  [main] Boundaries as shared edges: replaced {n_replaced} thin polygons", flush=True)
    return gpd.GeoDataFrame(geometry=parcels_list, crs=gdf.crs)


def map_to_pixel(x: float, y: float, minx: float, miny: float, maxx: float, maxy: float, width: int, height: int) -> tuple[int, int]:
    """Map coords (EPSG:25833) to pixel (i, j). i=column, j=row; row 0 = north (maxy)."""
    i = int((x - minx) / (maxx - minx) * (width - 1)) if maxx != minx else 0
    j = int((maxy - y) / (maxy - miny) * (height - 1)) if maxy != miny else 0
    i = max(0, min(width - 1, i))
    j = max(0, min(height - 1, j))
    return i, j


# Resolved GetFeatureInfo format: 'gml' or 'text', set on first call when format is auto
_getfeatureinfo_format: str | None = None


def parse_feature_info_gml(xml_text: str) -> dict[str, str]:
    """Parse GetFeatureInfo application/vnd.ogc.gml response into key-value dict."""
    out: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
        # Skip OGC ServiceExceptionReport (error response)
        tag_root = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if "Exception" in tag_root or "ServiceException" in tag_root:
            return out
        # Find feature element: *_feature or element with simple child tags (id, objid, nut, ...)
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag.endswith("_feature") or "feature" in tag.lower():
                for child in elem:
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if ctag not in ("boundedBy", "coordinates", "Box", "name"):
                        out[ctag] = (child.text or "").strip()
                if out:
                    return out
        # No *_feature wrapper: collect any simple key-like child elements of root
        for layer in root:
            for child in layer:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag not in ("boundedBy", "coordinates", "Box", "name"):
                    out[ctag] = (child.text or "").strip()
            if out:
                return out
    except ET.ParseError:
        pass
    return out


def get_feature_info(
    minx: float, miny: float, maxx: float, maxy: float,
    width: int, height: int, px: float, py: float,
    *,
    wms_url: str | None = None,
    wms_layer: str | None = None,
    crs: str | None = None,
    info_format: str | None = None,
) -> dict[str, str]:
    """Query GetFeatureInfo at map point (px, py). Returns dict of attribute name -> value.
    info_format: 'gml' | 'text' | None. None = try GML first, then text (and cache for subsequent calls).
    """
    global _getfeatureinfo_format
    wms_url = wms_url or DEFAULT_WMS_URL
    wms_layer = wms_layer or DEFAULT_WMS_LAYER
    crs = crs or DEFAULT_CRS
    i, j = map_to_pixel(px, py, minx, miny, maxx, maxy, width, height)
    fmt = info_format if info_format is not None else _getfeatureinfo_format

    if fmt == "gml":
        params = {
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
            "LAYERS": wms_layer, "STYLES": "default", "QUERY_LAYERS": wms_layer,
            "CRS": crs, "BBOX": f"{minx},{miny},{maxx},{maxy}",
            "WIDTH": width, "HEIGHT": height, "I": i, "J": j,
            "INFO_FORMAT": "application/vnd.ogc.gml", "FEATURE_COUNT": 1,
        }
        r = requests.get(wms_url, params=params, timeout=15)
        r.raise_for_status()
        return parse_feature_info_gml(r.text)

    if fmt == "text":
        params = {
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
            "LAYERS": wms_layer, "STYLES": "default", "QUERY_LAYERS": wms_layer,
            "CRS": crs, "BBOX": f"{minx},{miny},{maxx},{maxy}",
            "WIDTH": width, "HEIGHT": height, "I": i, "J": j,
            "INFO_FORMAT": "text/plain", "FEATURE_COUNT": 1,
        }
        r = requests.get(wms_url, params=params, timeout=15)
        r.raise_for_status()
        if _getfeatureinfo_format is None:
            _getfeatureinfo_format = "text"
            print(f"  [GetFeatureInfo] auto: using text/plain fallback", flush=True)
        return parse_feature_info_text(r.text)

    # Auto: try GML first
    params_gml = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
        "LAYERS": wms_layer, "STYLES": "default", "QUERY_LAYERS": wms_layer,
        "CRS": crs, "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width, "HEIGHT": height, "I": i, "J": j,
        "INFO_FORMAT": "application/vnd.ogc.gml", "FEATURE_COUNT": 1,
    }
    r = requests.get(wms_url, params=params_gml, timeout=15)
    if _getfeatureinfo_format is None:
        print(f"  [GetFeatureInfo] auto: tried GML, status={r.status_code}, len={len(r.text)}, ct={r.headers.get('Content-Type', '')[:50]}", flush=True)
    if r.status_code == 200 and (r.text.strip().startswith("<?xml") or "application/vnd.ogc.gml" in (r.headers.get("Content-Type") or "")):
        parsed = parse_feature_info_gml(r.text)
        if _getfeatureinfo_format is None:
            print(f"  [GetFeatureInfo] GML parsed keys: {list(parsed.keys()) if parsed else 'EMPTY'}", flush=True)
        if parsed:
            if _getfeatureinfo_format is None:
                _getfeatureinfo_format = "gml"
            return parsed
    # Fallback to text/plain
    params_txt = {**params_gml, "INFO_FORMAT": "text/plain"}
    r = requests.get(wms_url, params=params_txt, timeout=15)
    r.raise_for_status()
    if _getfeatureinfo_format is None:
        _getfeatureinfo_format = "text"
    return parse_feature_info_text(r.text)


def _interior_point(geom) -> tuple[float, float] | None:
    """
    Return (x, y) of a point guaranteed to be inside the polygon, for GetFeatureInfo queries.
    Centroid can lie outside for concave/L-shaped polygons; representative_point() is inside.
    """
    if geom is None or geom.is_empty:
        return None
    try:
        if geom.geom_type == "Polygon":
            pt = geom.representative_point()
        elif geom.geom_type == "MultiPolygon":
            # Use largest polygon part so the query hits the main area
            pt = max(geom.geoms, key=lambda p: p.area).representative_point()
        else:
            pt = geom.centroid
        if pt.is_empty:
            return None
        return (float(pt.x), float(pt.y))
    except Exception:
        return None


def parse_feature_info_text(text: str) -> dict[str, str]:
    """Parse GetFeatureInfo text/plain into key-value dict. Handles multiple formats for robustness:
    - key: value (first colon)
    - key = 'value' or key = \"value\" (MapServer)
    - key = value (unquoted; value = rest of line)
    - Tab-separated key\\tvalue
    - HTML <td>key</td><td>value</td>
    Skips lines whose key starts with 'Feature ' or 'Layer '.
    """
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip section headers
        if line.startswith("Feature ") or line.startswith("Layer "):
            continue
        # 1) key = 'value' or key = "value" (MapServer; quoted value)
        m = re.match(r"^([\w.]+)\s*=\s*['\"]([^'\"]*)['\"]", line)
        if m:
            out[m.group(1)] = m.group(2)
            continue
        # 2) key = value (unquoted; value = rest of line)
        m = re.match(r"^([\w.]+)\s*=\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
            continue
        # 3) key: value (first colon; value may contain colons)
        if ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key and not key.startswith("Feature ") and not key.startswith("Layer "):
                out[key] = val
            continue
        # 4) Tab-separated key\tvalue
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                out[parts[0].strip()] = parts[1].strip()
            continue
        # 5) HTML table row: <td>key</td><td>value</td>
        if "<td>" in line.lower():
            cells = re.findall(r"<td[^>]*>([^<]*)</td>", line, re.I)
            if len(cells) >= 2 and cells[0].strip():
                out[cells[0].strip()] = cells[1].strip()
    return out


def run(
    extent: tuple[float, float, float, float] | None = None,
    output_path: Path | str | None = None,
    *,
    input_boundary: Path | str | None = None,
    boundary_buffer_m: float | None = None,
    wms_url: str | None = None,
    wms_layer: str | None = None,
    crs: str | None = None,
    map_size: int | None = None,
    cache_dir: Path | str | None = None,
    line_color: str | None = None,
    line_color_rgb: tuple[int, int, int, int, int, int] | None = None,
    background_color: tuple[int, int, int] | None = None,
    background_tolerance: int | None = None,
    line_channel: str | None = None,
    line_channel_min: int | None = None,
    line_channel_max: int | None = None,
    max_hole_area_sq_m: float | None = None,
    download_only: bool | None = None,
    skip_getfeatureinfo: bool | None = None,
    getfeatureinfo_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Download WMS raster for extent, segment line boundaries (by color or channel), vectorize to parcel polygons,
    optionally fill interior rings and query GetFeatureInfo, write shapefile.

    Args:
        extent: (minx, miny, maxx, maxy) in crs. If None, required: input_boundary (bounds taken from that shape).
        output_path: Path for output shapefile. Written unless download_only=True. Default from DEFAULT_OUTPUT_SHP when extent from input_boundary.
        input_boundary: Shapefile path for extent when extent is None. Required if extent is None. Ignored if extent is given.
        boundary_buffer_m: Buffer in metres around input_boundary to expand extent. 0 = no buffer. Used only when extent is None.
        wms_url: WMS GetMap/GetFeatureInfo URL. Default: DEFAULT_WMS_URL.
        wms_layer: WMS layer name. Default: DEFAULT_WMS_LAYER.
        crs: CRS for bbox and output (e.g. EPSG:25833). Default: DEFAULT_CRS.
        map_size: Pixels on longest side; None = auto from GetCapabilities. Default: None.
        cache_dir: Directory for cached rasters. Default: project cache dir.
        line_color: Preset for line pixels: green|black|red|blue|white. Required unless line_color_rgb or line_channel is set (no default in library).
        line_color_rgb: Custom RGB range (r_min,r_max,g_min,g_max,b_min,b_max). Overrides line_color if set.
        background_color: (r,g,b); if set, line = pixel not within tolerance of this color.
        background_tolerance: Max distance from background_color to count as background. Default: 15.
        line_channel: Use single channel: alpha|0|1|2. If set, ignores line_color/line_color_rgb.
        line_channel_min, line_channel_max: Channel value range for line (e.g. alpha 128–255, black 0–80).
        max_hole_area_sq_m: None = no fill. 0 = fill all holes; >0 = fill holes smaller than this (m²).
        download_only: If True, only download/cache raster. None = use module default.
        skip_getfeatureinfo: If True, do not query GetFeatureInfo. None = use module default.
        getfeatureinfo_workers: Parallel HTTP requests for GetFeatureInfo. 1 = sequential (with delay). None = use GFI_WORKERS (default 8).

    Returns:
        Final parcel GeoDataFrame. Empty if download_only=True.
    """
    _crs = crs or DEFAULT_CRS
    if line_color is None and line_color_rgb is None and line_channel is None:
        raise ValueError("One of line_color, line_color_rgb, or line_channel must be provided")
    if extent is None:
        if not input_boundary:
            raise ValueError("Either extent or input_boundary must be provided")
        _buffer_m = boundary_buffer_m if boundary_buffer_m is not None else 0.0
        extent = get_bounds_from_shape(input_boundary, crs=_crs, buffer_m=_buffer_m)
        if output_path is None:
            output_path = DEFAULT_OUTPUT_SHP
    _out = output_path or DEFAULT_OUTPUT_SHP
    if not _out and (DOWNLOAD_ONLY if download_only is None else download_only) is False:
        raise ValueError("output_path is required when not download_only")
    output_path = Path(_out) if _out else None
    minx, miny, maxx, maxy = extent

    _max_hole = MAX_HOLE_AREA_SQ_M if max_hole_area_sq_m is None else max_hole_area_sq_m
    _download_only = DOWNLOAD_ONLY if download_only is None else download_only
    _skip_gfi = SKIP_GETFEATUREINFO if skip_getfeatureinfo is None else skip_getfeatureinfo
    _gfi_workers = GFI_WORKERS if getfeatureinfo_workers is None else getfeatureinfo_workers
    _cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

    print("[run] Starting...", flush=True)
    print(f"[run] Extent ({_crs}): {extent}", flush=True)

    print("[run] Downloading WMS raster (or loading from cache)...", flush=True)
    raster_path = download_wms_raster(
        minx, miny, maxx, maxy,
        wms_url=wms_url,
        wms_layer=wms_layer,
        crs=_crs,
        map_size=map_size,
        cache_dir=_cache_dir,
        cache_key_prefix="wms",
    )

    if _download_only:
        print("[run] download_only=True — skipping segmentation and export.", flush=True)
        return gpd.GeoDataFrame(geometry=[], crs=_crs)

    print("[run] Opening raster for extent...", flush=True)
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform
    minx_r, miny_r = transform * (0, height)
    maxx_r, maxy_r = transform * (width, 0)

    t_main = time.perf_counter()
    print("[run] Segmenting image and vectorizing polygons (boundary-based)...", flush=True)
    polygons_gdf = segment_and_vectorize(
        raster_path,
        line_color=line_color,
        line_color_rgb=line_color_rgb,
        background_color=background_color,
        background_tolerance=background_tolerance,
        line_channel=line_channel,
        line_channel_min=line_channel_min,
        line_channel_max=line_channel_max,
    )
    print(f"[run] Polygons from segmentation: {len(polygons_gdf)} ({(time.perf_counter() - t_main):.1f}s total)", flush=True)

    if "is_parcel" in polygons_gdf.columns:
        print("[run] Replacing thin 'line' polygons with shared edges...", flush=True)
        polygons_gdf = _boundaries_as_shared_edges(polygons_gdf)
        print(f"[run] Parcels after shared-edge step: {len(polygons_gdf)}", flush=True)
    else:
        print(f"[run] Split-extent path: boundaries are shared edges, {len(polygons_gdf)} parcels", flush=True)

    if SIMPLIFY_TOLERANCE_M > 0 and SIMPLIFY_BLADE_M <= 0 and len(polygons_gdf) > 0:
        polygons_gdf = polygons_gdf.copy()
        polygons_gdf["geometry"] = polygons_gdf.geometry.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
        polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty].copy()
        polygons_gdf = polygons_gdf[polygons_gdf.geometry.area >= MIN_PARCEL_AREA_M2].copy()
        print(f"[run] Simplified (tolerance={SIMPLIFY_TOLERANCE_M} m): {len(polygons_gdf)} parcels", flush=True)

    from shapely.geometry import box as shapely_box
    raster_bbox = shapely_box(minx_r, miny_r, maxx_r, maxy_r)
    t_main = time.perf_counter()
    print("[run] Clipping to WMS raster bbox and exploding multiparts...", flush=True)
    polygons_gdf["geometry"] = polygons_gdf.geometry.intersection(raster_bbox)
    polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty].copy()
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.area >= MIN_PARCEL_AREA_M2].copy()

    def extract_polygons(g):
        if g.geom_type == "Polygon" and not g.is_empty and g.area >= 1.0:
            return [g]
        if g.geom_type == "MultiPolygon":
            return [p for p in g.geoms if not p.is_empty and p.area >= 1.0]
        if g.geom_type == "GeometryCollection":
            out = []
            for x in g.geoms:
                out.extend(extract_polygons(x))
            return out
        return []

    all_polys = []
    for g in polygons_gdf.geometry:
        all_polys.extend(extract_polygons(g))
    polygons_gdf = gpd.GeoDataFrame(geometry=all_polys, crs=polygons_gdf.crs)
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.area >= MIN_PARCEL_AREA_M2].copy()
    print(f"[run] After clip to raster bbox: {len(polygons_gdf)} polygons ({(time.perf_counter() - t_main):.1f}s)", flush=True)

    if _max_hole is not None:
        def _fill_holes(geom):
            if geom.geom_type != "Polygon" or geom.is_empty or not geom.interiors:
                return geom
            if _max_hole <= 0:
                return Polygon(geom.exterior)
            keep = [r for r in geom.interiors if Polygon(r).area >= _max_hole]
            return Polygon(geom.exterior, keep) if keep else Polygon(geom.exterior)

        polygons_gdf = polygons_gdf.copy()
        polygons_gdf["geometry"] = polygons_gdf.geometry.apply(_fill_holes)
        msg = "all interior rings filled" if _max_hole <= 0 else f"holes < {_max_hole} m² filled"
        print(f"[run] {msg}", flush=True)

    if not _skip_gfi:
        global _getfeatureinfo_format
        _getfeatureinfo_format = None  # auto-detect GML vs text on first request
        n_poly = len(polygons_gdf)
        # Build (index, cx, cy) for polygons with interior point; preallocate attrs_list
        attrs_list = [{}] * n_poly
        tasks = []
        for i in range(n_poly):
            pt = _interior_point(polygons_gdf.iloc[i].geometry)
            if pt is None:
                continue
            tasks.append((i, pt[0], pt[1]))

        if _gfi_workers <= 1:
            print("[run] Querying GetFeatureInfo for each polygon (sequential)...", flush=True)
            for i, cx, cy in tasks:
                time.sleep(REQUEST_DELAY_S)
                try:
                    attrs = get_feature_info(
                        minx_r, miny_r, maxx_r, maxy_r, width, height, cx, cy,
                        wms_url=wms_url, wms_layer=wms_layer, crs=_crs,
                    )
                except Exception as e:
                    attrs = {"_error": str(e)}
                    if i == tasks[0][0]:
                        print(f"  [run] GetFeatureInfo ERROR on first polygon: {e!r}", flush=True)
                attrs_list[i] = attrs
                done = sum(1 for j in range(n_poly) if attrs_list[j])
                if done % 50 == 0 or done == len(tasks):
                    print(f"  [run] GetFeatureInfo: {done}/{n_poly} polygons", flush=True)
        else:
            print(f"[run] Querying GetFeatureInfo for each polygon ({_gfi_workers} workers)...", flush=True)

            def _gfi_one(item: tuple[int, float, float]) -> tuple[int, dict]:
                ii, cx, cy = item
                try:
                    return ii, get_feature_info(
                        minx_r, miny_r, maxx_r, maxy_r, width, height, cx, cy,
                        wms_url=wms_url, wms_layer=wms_layer, crs=_crs,
                    )
                except Exception as e:
                    return ii, {"_error": str(e)}

            # First request in main thread to set _getfeatureinfo_format
            if tasks:
                i0, cx0, cy0 = tasks[0]
                try:
                    attrs_list[i0] = get_feature_info(
                        minx_r, miny_r, maxx_r, maxy_r, width, height, cx0, cy0,
                        wms_url=wms_url, wms_layer=wms_layer, crs=_crs,
                    )
                except Exception as e:
                    attrs_list[i0] = {"_error": str(e)}
                    print(f"  [run] GetFeatureInfo ERROR on first polygon: {e!r}", flush=True)
                done_count = 1
                if done_count % 50 == 0 or done_count == len(tasks):
                    print(f"  [run] GetFeatureInfo: {done_count}/{n_poly} polygons", flush=True)

            # Rest in parallel
            if len(tasks) > 1:
                with ThreadPoolExecutor(max_workers=_gfi_workers) as ex:
                    futures = {ex.submit(_gfi_one, t): t for t in tasks[1:]}
                    for fut in as_completed(futures):
                        ii, attrs = fut.result()
                        attrs_list[ii] = attrs
                        done_count = sum(1 for a in attrs_list if a)
                        if done_count % 50 == 0 or done_count == n_poly:
                            print(f"  [run] GetFeatureInfo: {done_count}/{n_poly} polygons", flush=True)
            print(f"  [run] GetFeatureInfo: {len(tasks)}/{n_poly} polygons", flush=True)

        print("[run] Building attribute columns...", flush=True)
        all_keys = set()
        for a in attrs_list:
            all_keys.update(a.keys())
        non_underscore = sorted(k for k in all_keys if not k.startswith("_"))
        print(f"  [run] GetFeatureInfo attribute keys ({len(non_underscore)}): {non_underscore[:15]}{'...' if len(non_underscore) > 15 else ''}", flush=True)
        if not non_underscore and attrs_list:
            sample = attrs_list[0]
            print(f"  [run] WARNING: no attribute columns (first attrs keys: {list(sample.keys())})", flush=True)
        for k in sorted(all_keys):
            if k.startswith("_"):
                continue
            polygons_gdf[k] = [a.get(k, "") for a in attrs_list]
    else:
        print("[run] skip_getfeatureinfo=True — writing shapefile with geometry only.", flush=True)
        polygons_gdf["id"] = range(1, len(polygons_gdf) + 1)

    print("[run] Writing shapefile...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polygons_gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"[run] Written: {output_path}", flush=True)
    return polygons_gdf


def main() -> None:
    ap = argparse.ArgumentParser(description="WMS to shapefile: download layer, vectorize line boundaries to polygons")
    ap.add_argument("--input-boundary", type=Path, default=None, metavar="PATH",
        help=f"Boundary shapefile for extent (default: {DEFAULT_INPUT_BOUNDARY})")
    ap.add_argument("--boundary-buffer", type=float, default=None, metavar="M",
        help="Buffer in metres around boundary to expand extent (default: 0)")
    ap.add_argument("--output", "-o", type=Path, default=None, metavar="PATH",
        help=f"Output shapefile path (default: {DEFAULT_OUTPUT_SHP})")
    ap.add_argument("--wms-url", default=None, metavar="URL",
        help=f"WMS GetMap URL (default: {DEFAULT_WMS_URL})")
    ap.add_argument("--layer", default=None, dest="wms_layer", metavar="NAME",
        help=f"WMS layer name (default: {DEFAULT_WMS_LAYER})")
    ap.add_argument("--crs", default=None, help=f"CRS for bbox and output (default: {DEFAULT_CRS})")
    ap.add_argument("--map-size", default=None, metavar="N|auto",
        help="Pixels on longest side; 'auto' or omit = from GetCapabilities (default: auto)")
    ap.add_argument("--max-hole-area", metavar="M2|none", default=None,
        help="Fill interior rings: none=off, 0 or all=fill all, or number (m²) (default: use script constant)")
    ap.add_argument("--line-color", default=None, choices=["green", "black", "red", "blue", "white"],
        help="Line color preset (default: green)")
    ap.add_argument("--line-color-rgb", default=None, metavar="r_min,r_max,g_min,g_max,b_min,b_max",
        help="Custom RGB range for line pixels (0-255 each); overrides --line-color")
    ap.add_argument("--background-color", default=None, metavar="r,g,b",
        help="Background color (0-255); line = pixel not within --background-tolerance of this")
    ap.add_argument("--background-tolerance", type=int, default=None, metavar="N",
        help="Max distance from background to count as background (default: 15)")
    ap.add_argument("--line-channel", default=None, choices=["alpha", "0", "1", "2"],
        help="Use single channel for line: alpha or R=0,G=1,B=2; overrides color")
    ap.add_argument("--line-channel-range", default=None, metavar="min,max",
        help="Channel value range for line (default: alpha 128,255; RGB 0,80)")
    gfi = ap.add_mutually_exclusive_group()
    gfi.add_argument("--skip-getfeatureinfo", action="store_true",
        help="Skip GetFeatureInfo (write geometry only, no WMS attributes)")
    gfi.add_argument("--getfeatureinfo", action="store_true", dest="do_getfeatureinfo",
        help="Query GetFeatureInfo to get WMS attribute data (default when omitted: use script constant)")
    args = ap.parse_args()

    input_boundary = args.input_boundary or DEFAULT_INPUT_BOUNDARY
    output_path = args.output or DEFAULT_OUTPUT_SHP
    if not input_boundary or not output_path:
        print("[main] --input-boundary and --output are required (no project defaults in gis_utils).", file=sys.stderr, flush=True)
        sys.exit(1)
    line_color_val = args.line_color or DEFAULT_LINE_COLOR  # CLI default; library requires explicit

    line_color_rgb_val = None
    if args.line_color_rgb is not None:
        parts = [x.strip() for x in args.line_color_rgb.split(",")]
        if len(parts) != 6:
            print("[main] --line-color-rgb must be r_min,r_max,g_min,g_max,b_min,b_max (6 integers)", file=sys.stderr, flush=True)
            sys.exit(1)
        try:
            line_color_rgb_val = tuple(int(x) for x in parts)
        except ValueError:
            print("[main] --line-color-rgb values must be integers 0-255", file=sys.stderr, flush=True)
            sys.exit(1)

    background_color_val = None
    if args.background_color is not None:
        parts = [x.strip() for x in args.background_color.split(",")]
        if len(parts) != 3:
            print("[main] --background-color must be r,g,b (3 integers)", file=sys.stderr, flush=True)
            sys.exit(1)
        try:
            background_color_val = tuple(int(x) for x in parts)
        except ValueError:
            print("[main] --background-color values must be integers 0-255", file=sys.stderr, flush=True)
            sys.exit(1)

    line_channel_min_val = line_channel_max_val = None
    if args.line_channel_range is not None:
        parts = [x.strip() for x in args.line_channel_range.split(",")]
        if len(parts) != 2:
            print("[main] --line-channel-range must be min,max (2 integers)", file=sys.stderr, flush=True)
            sys.exit(1)
        try:
            line_channel_min_val, line_channel_max_val = int(parts[0]), int(parts[1])
        except ValueError:
            print("[main] --line-channel-range values must be integers", file=sys.stderr, flush=True)
            sys.exit(1)

    map_size_val = DEFAULT_MAP_SIZE
    if args.map_size is not None:
        v = args.map_size.strip().lower()
        if v in ("auto", ""):
            map_size_val = None
        else:
            try:
                map_size_val = int(v)
            except ValueError:
                print(f"[main] Invalid --map-size '{args.map_size}'; use 'auto' or an integer.", file=sys.stderr, flush=True)
                sys.exit(1)

    skip_getfeatureinfo_val = None
    if args.do_getfeatureinfo:
        skip_getfeatureinfo_val = False
    elif args.skip_getfeatureinfo:
        skip_getfeatureinfo_val = True

    max_hole_area_sq_m = MAX_HOLE_AREA_SQ_M
    if args.max_hole_area is not None:
        v = args.max_hole_area.strip().lower()
        if v in ("none", "off", "no"):
            max_hole_area_sq_m = None
        elif v in ("0", "all"):
            max_hole_area_sq_m = 0.0
        else:
            try:
                max_hole_area_sq_m = float(v)
            except ValueError:
                print(f"[main] Invalid --max-hole-area '{args.max_hole_area}'; use none, 0, or a number (m²).", file=sys.stderr, flush=True)
                sys.exit(1)

    run(
        extent=None,
        output_path=output_path,
        input_boundary=input_boundary,
        boundary_buffer_m=args.boundary_buffer,
        wms_url=args.wms_url,
        wms_layer=args.wms_layer,
        crs=args.crs,
        map_size=map_size_val,
        line_color=line_color_val,
        line_color_rgb=line_color_rgb_val,
        background_color=background_color_val,
        background_tolerance=args.background_tolerance,
        line_channel=args.line_channel,
        line_channel_min=line_channel_min_val,
        line_channel_max=line_channel_max_val,
        max_hole_area_sq_m=max_hole_area_sq_m,
        skip_getfeatureinfo=skip_getfeatureinfo_val,
    )


def get_planungsbereich_bounds(buffer_m: float = 10.0) -> tuple[float, float, float, float]:
    """Backwards compatibility: bounds from DEFAULT_INPUT_BOUNDARY. Prefer get_bounds_from_shape(path, crs, buffer_m)."""
    if DEFAULT_INPUT_BOUNDARY is None:
        raise ValueError("DEFAULT_INPUT_BOUNDARY not set; use get_bounds_from_shape(path, crs, buffer_m) with explicit path.")
    return get_bounds_from_shape(DEFAULT_INPUT_BOUNDARY, DEFAULT_CRS, buffer_m)


if __name__ == "__main__":
    print("(script entry)", flush=True)
    main()

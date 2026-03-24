"""
Download polygon data from OpenStreetMap via the Overpass API.

Provides functions to query settlement areas, land use, and other polygon
features from OSM, with dissolve and morphological filtering support.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, polygonize, unary_union

from gis_utils.geometry import remove_inner_rings


# ---------------------------------------------------------------------------
# Overpass querying
# ---------------------------------------------------------------------------

DEFAULT_OVERPASS_URL = "http://overpass-api.de/api/interpreter"
CACHE_DIR_NAME = "download_cache"
DEFAULT_CACHE_MAX_AGE_S = 86400  # 24 hours


def _osm_cache_key(
    bbox: tuple[float, float, float, float],
    tags: dict[str, str],
    crs: str,
    dissolve: bool,
) -> str:
    """Build a deterministic cache filename from OSM query parameters."""
    parts = [
        f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}",
        crs,
        "dissolved" if dissolve else "raw",
    ]
    for k in sorted(tags):
        parts.append(f"{k}={tags[k]}")
    raw = "|".join(parts)
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"osm_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}_{short_hash}.gpkg"


def download_osm_polygons(
    bbox: tuple[float, float, float, float],
    tags: dict[str, str] | None = None,
    *,
    crs: str,
    dissolve: bool = True,
    timeout: int = 180,
    overpass_url: str = DEFAULT_OVERPASS_URL,
    cache_dir: Path | str | None = None,
    cache_max_age_s: int = DEFAULT_CACHE_MAX_AGE_S,
    no_cache: bool = False,
) -> gpd.GeoDataFrame:
    """
    Download polygon features from OpenStreetMap via the Overpass API.

    Args:
        bbox: Bounding box in WGS84 as (minx, miny, maxx, maxy) = (west, south, east, north).
        tags: Dict of OSM tag filters. Keys are tag names, values are regex patterns.
            Default: settlement areas (landuse=residential/commercial/industrial/retail,
            place=city/town/village/hamlet/suburb/neighbourhood).
        crs: Reproject results to this CRS. Must be projected (meters) for area calculation.
        dissolve: If True, dissolve touching/overlapping polygons into contiguous areas.
        timeout: Overpass API timeout in seconds.
        overpass_url: Overpass API endpoint.
        cache_dir: Directory for cached downloads. Default: download_cache/ in cwd.
        cache_max_age_s: Max age of cached file in seconds before re-downloading (default: 24h).
        no_cache: If True, skip cache and always download fresh.

    Returns:
        GeoDataFrame with polygon geometries, area_m2, and area_ha columns.
        If dissolve=True, attribute columns are dropped (meaningless after dissolve).
    """
    if tags is None:
        tags = {
            "landuse": "^(residential|commercial|industrial|retail)$",
            "place": "^(city|town|village|hamlet|suburb|neighbourhood)$",
        }

    # --- Cache check ---
    _cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / CACHE_DIR_NAME
    cache_file = _cache_dir / _osm_cache_key(bbox, tags, crs, dissolve)

    if not no_cache and cache_file.exists():
        age_s = time.time() - cache_file.stat().st_mtime
        if age_s < cache_max_age_s:
            print(f"[osm] Using cached data: {cache_file.name} ({age_s/3600:.1f}h old)", flush=True)
            return gpd.read_file(cache_file)

    minx, miny, maxx, maxy = bbox
    bbox_str = f"{miny},{minx},{maxy},{maxx}"

    # Build Overpass query
    query_parts = []
    for key, pattern in tags.items():
        query_parts.append(f'way["{key}"~"{pattern}"]({bbox_str});')
        query_parts.append(f'relation["{key}"~"{pattern}"]({bbox_str});')

    query = f"[out:json][timeout:{timeout}];(\n" + "\n".join(query_parts) + "\n);\nout geom;"

    print(f"Querying Overpass API...")
    response = requests.post(overpass_url, data={"data": query}, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    # Parse elements into features
    elements = data.get("elements", [])
    print(f"  Retrieved {len(elements)} elements")

    features = []
    for element in elements:
        tags_data = element.get("tags", {})
        geom = _parse_osm_element(element)
        if geom is None or geom.is_empty:
            continue
        features.append({
            "osm_id": element.get("id"),
            "osm_type": element.get("type"),
            "name": tags_data.get("name", ""),
            "landuse": tags_data.get("landuse", ""),
            "place": tags_data.get("place", ""),
            "geometry": geom,
        })

    if not features:
        print("  No valid geometries found")
        return gpd.GeoDataFrame(columns=["geometry", "area_m2", "area_ha"], crs=crs)

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326").to_crs(crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10_000
    print(f"  {len(gdf)} valid polygons")

    if dissolve:
        gdf = _dissolve_polygons(gdf)

    # Write to cache
    _cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_file, driver="GPKG")
    print(f"  Cached: {cache_file.name}", flush=True)

    return gdf


def bbox_from_shapefile(
    path: str | Path,
    crs: str = "EPSG:4326",
) -> tuple[float, float, float, float]:
    """
    Get bounding box from a shapefile, reprojected to the given CRS.

    Args:
        path: Path to shapefile.
        crs: Target CRS for the bounding box (default: WGS84).

    Returns:
        (minx, miny, maxx, maxy) in the target CRS.
    """
    gdf = gpd.read_file(path)
    gdf_reproj = gdf.to_crs(crs)
    return tuple(gdf_reproj.total_bounds)


# ---------------------------------------------------------------------------
# OSM geometry parsing
# ---------------------------------------------------------------------------

def _parse_osm_element(element: dict) -> Polygon | MultiPolygon | None:
    """Parse an Overpass element (way or relation) into a Shapely geometry."""
    elem_type = element.get("type")
    if elem_type == "way":
        return _build_polygon_from_way(element)
    if elem_type == "relation":
        return _build_polygon_from_relation(element)
    return None


def _build_polygon_from_way(element: dict) -> Polygon | None:
    if "geometry" not in element:
        return None
    coords = [(n["lon"], n["lat"]) for n in element["geometry"]]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    if len(coords) < 4:
        return None
    try:
        return Polygon(coords)
    except Exception:
        return None


def _build_polygon_from_relation(element: dict) -> Polygon | MultiPolygon | None:
    members = element.get("members", [])
    outer_lines, inner_lines = [], []

    for member in members:
        if member.get("type") != "way" or "geometry" not in member:
            continue
        coords = [(n["lon"], n["lat"]) for n in member["geometry"]]
        if len(coords) < 2:
            continue
        line = LineString(coords)
        role = member.get("role", "")
        if role == "outer":
            outer_lines.append(line)
        elif role == "inner":
            inner_lines.append(line)

    if not outer_lines:
        return None

    outer_poly = _lines_to_polygon(outer_lines)
    if outer_poly is None:
        return None

    # Subtract inner rings
    for hole in _lines_to_polygons_list(inner_lines):
        try:
            outer_poly = outer_poly.difference(hole)
        except Exception:
            pass

    if not outer_poly.is_valid:
        outer_poly = outer_poly.buffer(0)
    if outer_poly.is_empty:
        return None
    return outer_poly


def _lines_to_polygon(lines: list[LineString]) -> Polygon | MultiPolygon | None:
    """Merge lines and attempt to form a polygon."""
    merged = linemerge(lines)
    if isinstance(merged, LineString):
        coords = list(merged.coords)
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) >= 4:
                return Polygon(coords)
        return None

    if hasattr(merged, "geoms"):
        polys = list(polygonize(merged.geoms))
        if not polys:
            polys = _lines_to_polygons_list(list(merged.geoms))
        if polys:
            return unary_union(polys)
    return None


def _lines_to_polygons_list(lines: list[LineString]) -> list[Polygon]:
    """Try to form individual polygons from each line."""
    result = []
    for line in lines:
        coords = list(line.coords)
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) >= 4:
                try:
                    result.append(Polygon(coords))
                except Exception:
                    pass
    return result


# ---------------------------------------------------------------------------
# Dissolve and filter helpers
# ---------------------------------------------------------------------------

def _dissolve_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve touching/overlapping polygons."""
    dissolved = gdf.dissolve()
    if len(dissolved) == 1:
        dissolved = dissolved.explode(index_parts=False)
    dissolved = dissolved.reset_index(drop=True)
    dissolved["area_m2"] = dissolved.geometry.area
    dissolved["area_ha"] = dissolved["area_m2"] / 10_000
    # Keep only geometry + area columns (attributes meaningless after dissolve)
    return dissolved[["geometry", "area_m2", "area_ha"]].copy()

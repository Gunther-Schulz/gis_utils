"""
Geometry utilities for GIS workflows.

Provides common operations on Shapely geometries and GeoPandas GeoDataFrames:
polygon hole removal, geometry repair, set operations, and loading helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid as _shapely_make_valid


def remove_inner_rings(geom) -> Any:
    """
    Remove all inner rings (holes) from a geometry.

    Works on Polygon, MultiPolygon, and GeometryCollection. Other geometry
    types and None/empty values pass through unchanged.

    Args:
        geom: A Shapely geometry.

    Returns:
        Geometry with holes removed.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior) if geom.interiors else geom
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([remove_inner_rings(p) for p in geom.geoms])
    if geom.geom_type == "GeometryCollection":
        from shapely.geometry import GeometryCollection
        return GeometryCollection([remove_inner_rings(g) for g in geom.geoms])
    return geom


def make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Attempt to fix all invalid geometries in a GeoDataFrame.

    Non-destructive: returns a copy. Silently returns the original on error.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with repaired geometries.
    """
    try:
        out = gdf.copy()
        out["geometry"] = out.geometry.make_valid()
        return out
    except Exception:
        return gdf


def subtract_geometries(
    base_gdf: gpd.GeoDataFrame,
    subtract_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Compute set difference: base_gdf minus the union of subtract_gdf.

    Auto-reprojects subtract_gdf if CRS differs. Results contain only
    Polygon geometries (MultiPolygons are exploded).

    Args:
        base_gdf: GeoDataFrame to subtract from.
        subtract_gdf: GeoDataFrame whose union is subtracted.

    Returns:
        GeoDataFrame with Polygon geometries in base_gdf's CRS.
    """
    if subtract_gdf.empty:
        return base_gdf.copy()

    if base_gdf.crs and subtract_gdf.crs and base_gdf.crs != subtract_gdf.crs:
        subtract_gdf = subtract_gdf.to_crs(base_gdf.crs)

    sub_geoms = [_shapely_make_valid(g) for g in subtract_gdf.geometry if g is not None]
    sub_union = unary_union(sub_geoms)

    if sub_union.is_empty:
        return base_gdf.copy()

    result_geoms: list[Polygon] = []
    for _, row in base_gdf.iterrows():
        if row.geometry is None:
            continue
        geom = _shapely_make_valid(row.geometry)
        diff = geom.difference(sub_union)
        if diff.is_empty:
            continue
        for part in _extract_polygons(diff):
            result_geoms.append(part)

    if not result_geoms:
        return gpd.GeoDataFrame(columns=base_gdf.columns, crs=base_gdf.crs)

    return gpd.GeoDataFrame(geometry=result_geoms, crs=base_gdf.crs)


def subtract_smaller_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Where polygons overlap, subtract the smaller from the larger.

    Processes polygons in descending area order. For each polygon, subtracts
    all smaller intersecting polygons from it.

    Args:
        gdf: GeoDataFrame with polygon geometries.

    Returns:
        GeoDataFrame with non-overlapping polygons.
    """
    geoms = list(gdf.geometry)
    areas = [g.area if g is not None else 0 for g in geoms]
    n = len(geoms)
    by_area = sorted(range(n), key=lambda i: -areas[i])

    for j in by_area:
        if geoms[j] is None or geoms[j].is_empty:
            continue
        for i in range(n):
            if i == j or geoms[i] is None or geoms[i].is_empty:
                continue
            if areas[i] >= areas[j]:
                continue
            if not geoms[i].intersects(geoms[j]):
                continue
            try:
                new_j = geoms[j].difference(geoms[i])
                if new_j.is_empty:
                    geoms[j] = None
                    break
                geoms[j] = new_j
            except Exception:
                pass

    out = gdf.copy()
    out["geometry"] = geoms
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out.reset_index(drop=True)


def load_and_union(
    path: str | Path,
    crs: str | None = None,
) -> tuple[Any | None, gpd.GeoDataFrame | None]:
    """
    Load a shapefile and union all geometries to a single shape.

    Useful for avoiding double-counting overlapping polygons in area calculations.

    Args:
        path: Path to shapefile (or any format geopandas can read).
        crs: Reproject to this CRS. None = keep original.

    Returns:
        Tuple of (unioned_geometry, geodataframe), or (None, None) on error.
    """
    path = Path(path)
    if not path.exists():
        return None, None

    gdf = gpd.read_file(path)
    if gdf.empty:
        return None, None

    if crs:
        gdf = gdf.to_crs(crs)

    union_geom = gdf.union_all()
    return union_geom, gdf


def find_column(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    """
    Find the first column name from a list of candidates that exists in a GeoDataFrame.

    Useful for handling shapefiles with varying column naming conventions.

    Args:
        gdf: GeoDataFrame to search.
        candidates: Column name candidates in priority order.

    Returns:
        First matching column name, or None.
    """
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_polygons(geom) -> list[Polygon]:
    """Extract Polygon parts from any geometry type."""
    if geom.geom_type == "Polygon" and not geom.is_empty:
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [p for p in geom.geoms if not p.is_empty]
    if geom.geom_type == "GeometryCollection":
        out: list[Polygon] = []
        for g in geom.geoms:
            if g.geom_type == "Polygon" and not g.is_empty:
                out.append(g)
        return out
    return []

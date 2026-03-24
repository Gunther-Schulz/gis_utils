"""WFS download: fetch vector features directly from a WFS service.

Much simpler than WMS vectorization — no raster download, no color detection.
Use this when a WFS endpoint is available for the data source.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import geopandas as gpd

CACHE_DIR_NAME = "download_cache"


def _cache_key(layer: str, extent: tuple | None, crs: str, filter_hash: str = "") -> str:
    """Build a deterministic cache filename from request parameters."""
    parts = [layer, crs, filter_hash]
    if extent:
        parts.extend(f"{v:.0f}" for v in extent)
    raw = "_".join(parts)
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    safe_layer = layer.replace("/", "_").replace("\\", "_").replace(":", "_")
    if extent:
        minx, miny, maxx, maxy = extent
        return f"{safe_layer}_{minx:.0f}_{miny:.0f}_{maxx:.0f}_{maxy:.0f}_{short_hash}.gpkg"
    return f"{safe_layer}_{short_hash}.gpkg"


def download(
    url: str,
    layer: str,
    *,
    extent: tuple[float, float, float, float] | None = None,
    input_boundary: Path | str | None = None,
    output_path: Path | str | None = None,
    crs: str | None = None,
    version: str = "2.0.0",
    max_features: int | None = None,
    cache_dir: Path | str | None = None,
    no_cache: bool = False,
    recipe: "str | Recipe | None" = None,
    recipe_dir: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Download vector features from a WFS service.

    Args:
        url: WFS service URL.
        layer: Feature type name (e.g. 'adv:AX_Gebaeude').
        extent: (minx, miny, maxx, maxy) bounding box filter in crs.
        input_boundary: Shapefile/GeoPackage to derive extent from.
        output_path: Output file path (.gpkg or .shp). If None, no file written.
        crs: CRS for the request and output.
        version: WFS version (default '2.0.0').
        max_features: Limit number of features returned.
        cache_dir: Directory for cached downloads. Default: download_cache/ in cwd.
        no_cache: If True, skip cache and always download fresh.
        recipe: Recipe name or Recipe object for attribute mappings and post-processing.
        recipe_dir: Project directory for recipe search.

    Returns:
        GeoDataFrame with downloaded features.
    """
    # --- Recipe resolution ---
    _recipe = None
    if recipe is not None:
        from gis_utils.recipes import Recipe as _RecipeCls, load_recipe, resolve_connection
        if isinstance(recipe, str):
            _recipe = load_recipe(recipe, project_dir=Path(recipe_dir) if recipe_dir else None)
        else:
            _recipe = recipe
        _conn = resolve_connection(_recipe)
        url = url or _conn.get("wfs_url") or _conn.get("wms_url", "")
        layer = layer or _conn.get("layer", "")
        crs = crs or _conn.get("crs")

    if not crs:
        raise ValueError("crs is required (e.g. 'EPSG:25833'). No silent defaults — wrong CRS causes silent data corruption.")

    # Resolve extent from input_boundary
    if extent is None and input_boundary is not None:
        boundary_gdf = gpd.read_file(input_boundary)
        boundary_gdf = boundary_gdf.to_crs(crs)
        extent = tuple(boundary_gdf.total_bounds)

    # Build filter hash for cache key (includes exclude_tags so cache invalidates on filter change)
    filter_hash = ""
    if _recipe and _recipe.exclude_tags:
        filter_hash = hashlib.md5(str(sorted(_recipe.exclude_tags.items())).encode()).hexdigest()[:6]

    # --- Cache check ---
    _cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / CACHE_DIR_NAME
    cache_file = _cache_dir / _cache_key(layer, extent, crs, filter_hash)
    _cache_hit = False

    if not no_cache and cache_file.exists():
        gdf = gpd.read_file(cache_file)
        _cache_hit = True
    else:
        print(f"[wfs] Downloading {layer}...", flush=True)
        if extent:
            print(f"[wfs] Extent ({crs}): {extent[0]:.0f},{extent[1]:.0f} — {extent[2]:.0f},{extent[3]:.0f}", flush=True)

        # Use geopandas OGR WFS driver — handles complex GML (NAS, INSPIRE) reliably
        wfs_uri = f"WFS:{url}"
        read_kwargs = {"layer": layer}
        if extent:
            read_kwargs["bbox"] = extent
        if max_features:
            read_kwargs["max_features"] = max_features

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Field with same name")
            gdf = gpd.read_file(wfs_uri, **read_kwargs)

        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        else:
            gdf = gdf.to_crs(crs)

        print(f"[wfs] Downloaded {len(gdf)} features", flush=True)

        # Apply exclude_tags filter before caching
        if _recipe and _recipe.exclude_tags and len(gdf) > 0:
            drop_mask = None
            for col, pattern in _recipe.exclude_tags.items():
                if col not in gdf.columns:
                    continue
                col_match = gdf[col].fillna("").astype(str).str.match(pattern)
                drop_mask = col_match if drop_mask is None else (drop_mask | col_match)
            if drop_mask is not None:
                n_before = len(gdf)
                gdf = gdf[~drop_mask].copy()
                n_dropped = n_before - len(gdf)
                if n_dropped > 0:
                    print(f"[wfs] Excluded {n_dropped} features by exclude_tags filter", flush=True)

        _cache_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_file, driver="GPKG")

    # --- Recipe post-processing pipeline ---
    if _recipe is not None:
        from gis_utils.recipes import (
            apply_attribute_mappings,
            apply_column_mapping,
            apply_post_processing,
            load_and_run_hook,
        )
        _proj_dir = Path(recipe_dir) if recipe_dir else None

        if _recipe.attribute_mappings:
            apply_attribute_mappings(gdf, _recipe.attribute_mappings)
        if _recipe.post_processing:
            gdf = apply_post_processing(gdf, _recipe.post_processing)
        if _recipe.hooks:
            gdf = load_and_run_hook(_recipe.hooks, "post_process", gdf, _proj_dir)
        if _recipe.column_mapping:
            is_shp = output_path is not None and str(output_path).lower().endswith(".shp")
            gdf = apply_column_mapping(gdf, _recipe.column_mapping, is_shapefile=is_shp)

    # --- Write output ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext == ".shp":
            driver = "ESRI Shapefile"
        elif ext in (".gpkg", ".geopackage"):
            driver = "GPKG"
        elif ext == ".geojson":
            driver = "GeoJSON"
        else:
            driver = "GPKG"
        if not _cache_hit:
            print(f"[wfs] Writing output ({driver})...", flush=True)
        gdf.to_file(output_path, driver=driver)
        if not _cache_hit:
            print(f"[wfs] Written: {output_path}", flush=True)

    return gdf

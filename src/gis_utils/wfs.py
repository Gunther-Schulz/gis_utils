"""WFS download: fetch vector features directly from a WFS service.

Much simpler than WMS vectorization — no raster download, no color detection.
Use this when a WFS endpoint is available for the data source.
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import requests


def download(
    url: str,
    layer: str,
    *,
    extent: tuple[float, float, float, float] | None = None,
    input_boundary: Path | str | None = None,
    output_path: Path | str | None = None,
    crs: str | None = None,
    version: str = "1.1.0",
    max_features: int | None = None,
    recipe: "str | Recipe | None" = None,
    recipe_dir: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Download vector features from a WFS service.

    Args:
        url: WFS service URL.
        layer: Feature type name (e.g. 't7_moor_kbk25').
        extent: (minx, miny, maxx, maxy) bounding box filter in crs.
        input_boundary: Shapefile/GeoPackage to derive extent from.
        output_path: Output file path (.gpkg or .shp). If None, no file written.
        crs: CRS for the request and output.
        version: WFS version (default '1.1.0').
        max_features: Limit number of features returned.
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

    print(f"[wfs] Downloading features from {layer}...", flush=True)
    if extent:
        print(f"[wfs] Extent ({crs}): {extent}", flush=True)

    params = {
        "SERVICE": "WFS",
        "VERSION": version,
        "REQUEST": "GetFeature",
        "TYPENAME": layer,
        "SRSNAME": crs,
    }
    if extent:
        minx, miny, maxx, maxy = extent
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy},{crs}"
    if max_features:
        params["MAXFEATURES"] = str(max_features)

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()

    if b"Exception" in r.content:
        raise RuntimeError(f"WFS error: {r.text[:500]}")

    gdf = gpd.read_file(io.BytesIO(r.content))
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        gdf = gdf.to_crs(crs)

    print(f"[wfs] Downloaded {len(gdf)} features", flush=True)

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
            print("[wfs] Applying attribute mappings...", flush=True)
            apply_attribute_mappings(gdf, _recipe.attribute_mappings)

        if _recipe.post_processing:
            print("[wfs] Applying post-processing steps...", flush=True)
            gdf = apply_post_processing(gdf, _recipe.post_processing)

        if _recipe.hooks:
            print(f"[wfs] Running post-process hook: {_recipe.hooks}", flush=True)
            gdf = load_and_run_hook(_recipe.hooks, "post_process", gdf, _proj_dir)

        if _recipe.column_mapping:
            is_shp = output_path is not None and str(output_path).lower().endswith(".shp")
            print("[wfs] Applying column mapping...", flush=True)
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
        print(f"[wfs] Writing output ({driver})...", flush=True)
        gdf.to_file(output_path, driver=driver)
        print(f"[wfs] Written: {output_path}", flush=True)

    return gdf

"""concentric_point_buffers — Create concentric buffer rings around points.

For each point feature, creates multiple concentric buffer polygons based on
a lookup table of radii per category (e.g. species-specific protection zones
from BNatSchG).  Each buffer ring carries zone and radius attributes for
styling with graduated transparency or hatching in QGIS.

The lookup table maps a category value (e.g. bird species name) to one or
more named zones with radii.  Points whose category is not in the lookup
table are skipped.

Example workflow.yaml::

    - name: Nest Schutzzonen
      template: concentric_point_buffers
      params:
        input: GIS/Nester_2025.gpkg
        crs: "EPSG:25832"
        category_col: Art
        zones:
          - name: Nahbereich
            radius_col: Nahbereich_m
          - name: Zentraler Prüfbereich
            radius_col: Zentraler_Prüfbereich_m
          - name: Erweiterter Prüfbereich
            radius_col: Erweiterter_Prüfbereich_m
        lookup_csv: Grundlagen/BNatSchG_Brutvogelarten.csv
        lookup_key: Brutvogelarten
      output: output/nest_schutzzonen.gpkg

If no lookup_csv is given, the zone radii are taken directly from the
input features (the radius_col must exist in the input GeoDataFrame).
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "concentric_point_buffers",
    description="Create concentric buffer rings around points by category (e.g. BNatSchG zones)",
    params=["input", "crs", "category_col", "zones", "lookup_csv", "lookup_key"],
)
def concentric_point_buffers(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
    """Create concentric buffer zones around point features.

    Params:
        input: Path to GeoPackage/shapefile with point geometry.
        crs: Projected CRS for buffering.
        category_col: Column in input that identifies the category
            (e.g. ``"Art"`` for bird species).
        zones: List of zone definitions, each with:
            - ``name``: zone label (e.g. ``"Nahbereich"``)
            - ``radius_col``: column name in lookup table (or input) with radius
        lookup_csv (optional): Path to CSV lookup table mapping category
            values to zone radii.
        lookup_key (optional): Column name in the CSV that matches
            ``category_col`` values.  Required if ``lookup_csv`` is given.
    """
    import pandas as pd
    import geopandas as gpd

    input_path = project_dir / params["input"]
    crs = params["crs"]
    category_col = params["category_col"]
    zones = params["zones"]
    lookup_csv = params.get("lookup_csv")
    lookup_key = params.get("lookup_key")

    gdf = gpd.read_file(input_path).to_crs(crs)

    # Build lookup dict: category_value -> {radius_col: radius_value}
    lookup = {}
    if lookup_csv:
        csv_path = project_dir / lookup_csv
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = row[lookup_key]
            lookup[key] = {col: row.get(col) for col in df.columns}

    all_buffers = []
    for _, row in gdf.iterrows():
        category = row.get(category_col, "")

        # Get radii either from lookup or from the feature itself
        if lookup:
            if category not in lookup:
                continue
            radius_source = lookup[category]
        else:
            radius_source = row

        for zone_cfg in zones:
            zone_name = zone_cfg["name"]
            radius_col = zone_cfg["radius_col"]

            radius = radius_source.get(radius_col, 0)
            if pd.isna(radius) or radius is None or radius <= 0:
                continue

            buf = row.geometry.buffer(float(radius))
            all_buffers.append({
                "geometry": buf,
                category_col: category,
                "zone": zone_name,
                "radius_m": float(radius),
            })

    if not all_buffers:
        print("  No buffer zones created (no matching categories)")
        return False

    result = gpd.GeoDataFrame(all_buffers, crs=crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_path, driver="GPKG")
    print(f"  Concentric buffers: {len(result)} zones from {len(gdf)} points")
    return True

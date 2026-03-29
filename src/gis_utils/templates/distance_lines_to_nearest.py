"""distance_lines_to_nearest — Create distance LineStrings to nearest features.

For each feature in one or more reference layers, creates a LineString from
a target geometry (unioned) to the nearest point on the reference feature.
Output includes distance attributes and a formatted label for map styling.

The resulting LineStrings can be styled as arrows in QGIS to show distances
from a project site (WEA, BHKW, PV-Anlage, etc.) to nearby Schutzgebiete,
Biotope, or other reference areas.

Each output feature has:
- ``type``: reference layer type label
- ``name``: feature name from the reference layer
- ``dist_m``: distance in metres (rounded to 1 decimal)
- ``dist_label``: German-formatted distance string (e.g. "1.234 m")

Example workflow.yaml::

    - name: Schutzgebiete Distanzen
      template: distance_lines_to_nearest
      params:
        target: Geodaten/BHKW/BHKW_Gesamt.gpkg
        references:
          - file: Geodaten/Schutzgebiete/natura2000.gpkg
            name_col: gebietsnam
            type: Natura 2000
          - file: Geodaten/Schutzgebiete/biotopverbund.gpkg
            name_col: name
            type: Biotopverbund
        crs: "EPSG:25832"
        extent: [535812, 5997092, 538040, 5998666]
      output: output/distanzlinien.gpkg
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


def _fmt_de(val: float) -> str:
    """Format a number with German locale (dot as thousands separator)."""
    return f"{val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


@register(
    "distance_lines_to_nearest",
    description="Create distance LineStrings from target to nearest reference features (styleable as arrows)",
    params=["target", "references", "crs", "extent"],
)
def distance_lines_to_nearest(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
    """Create distance lines from a target to nearest points on reference features.

    Params:
        target: Path to GeoPackage/shapefile with the target geometry (e.g. WEA
            locations).  All features are unioned into a single geometry for
            distance measurement.
        references: List of reference layer configs, each with:
            - ``file``: path to GeoPackage/shapefile (relative to project root)
            - ``name_col``: column containing the feature name for labelling
            - ``type``: label for this reference category (e.g. "Natura 2000")
        crs: Projected CRS for distance calculation (e.g. ``"EPSG:25832"``).
        extent (optional): Bounding box ``[xmin, ymin, xmax, ymax]`` to filter
            reference features.  Only features intersecting this box are included.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, box
    from shapely.ops import nearest_points, unary_union

    target_path = project_dir / params["target"]
    references = params["references"]
    crs = params["crs"]
    extent = params.get("extent")

    extent_geom = box(*extent) if extent else None

    # Load and union target geometry
    target_gdf = gpd.read_file(target_path).to_crs(crs)
    target_geom = unary_union(target_gdf.geometry)

    all_lines = []
    for ref_cfg in references:
        ref_path = project_dir / ref_cfg["file"]
        name_col = ref_cfg.get("name_col")
        ref_type = ref_cfg.get("type", ref_path.stem)

        gdf = gpd.read_file(ref_path).to_crs(crs)

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if extent_geom and not geom.intersects(extent_geom):
                continue

            dist = target_geom.distance(geom)

            name = ""
            if name_col and name_col in gdf.columns:
                val = row.get(name_col)
                if isinstance(val, str):
                    name = val.strip()
            if not name:
                name = f"{ref_type} #{i}"

            p_target, p_ref = nearest_points(target_geom, geom)
            line = LineString([p_target, p_ref])

            all_lines.append({
                "geometry": line,
                "type": ref_type,
                "name": name,
                "dist_m": round(dist, 1),
                "dist_label": f"{_fmt_de(round(dist))} m",
            })

    if not all_lines:
        print("  No reference features found within extent")
        return False

    gdf_lines = gpd.GeoDataFrame(all_lines, crs=crs)
    gdf_lines = gdf_lines.sort_values("dist_m").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_lines.to_file(output_path, driver="GPKG")
    print(f"  Distance lines: {len(gdf_lines)} features -> {output_path}")
    return True

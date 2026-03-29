"""verification_dxf — Write original input + derived output to DXF for visual QA.

Creates a DXF file with two layers so you can overlay the original input
geometry and the derived result in AutoCAD or QGIS to verify accuracy:

- **ORIGINAL_{layer}** (red, color 1): the raw input lines from the DXF
- **DERIVED_Grenze** (green, color 3): the polygon boundary we derived

Use when: You want to visually confirm that a derived polygon (from
``dxf_lines_to_polygon`` or similar) faithfully follows the original geometry.

Example workflow.yaml::

    - name: Verification DXF
      template: verification_dxf
      params:
        dxf: Grundlagen/entwurf.dxf
        layer: PL_LIN_Materialwechsel
        crs: "EPSG:25833"
        strip_zone: true
        polygon: output/grenze.gpkg
      output: output/verification.dxf
      run: always
      depends_on:
        - Materialwechsel Grenze
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "verification_dxf",
    description="Write original DXF lines + derived polygon to DXF for visual QA",
    params=["dxf", "layer", "crs", "strip_zone", "polygon"],
)
def verification_dxf(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
    """Create a QA DXF overlaying input lines and derived polygon.

    Params:
        dxf: Path to the original DXF file (relative to project root).
        layer: DXF layer name containing the input lines.
        crs: Coordinate reference system.
        strip_zone (optional): Strip UTM zone prefix.  Default ``false``.
        polygon: Path to the derived polygon GeoPackage (relative to project root).
    """
    import geopandas as gpd

    from gis_utils import (
        ensure_layer,
        extract_dxf_layers,
        new_dxf_document,
        strip_utm_zone_prefix,
    )

    dxf_path = project_dir / params["dxf"]
    layer = params["layer"]
    crs = params["crs"]
    strip_zone = params.get("strip_zone", False)
    polygon_path = project_dir / params["polygon"]

    # Load original lines
    layers = extract_dxf_layers(dxf_path, crs=crs, layers=[layer])
    lines_gdf = layers[layer]["LineString"]
    if strip_zone:
        lines_gdf = strip_utm_zone_prefix(lines_gdf)

    # Load derived polygon
    polygon_gdf = gpd.read_file(polygon_path)

    # Build DXF
    doc = new_dxf_document()
    msp = doc.modelspace()

    ensure_layer(doc, f"ORIGINAL_{layer}", color=1)  # red
    ensure_layer(doc, "DERIVED_Grenze", color=3)  # green

    for geom in lines_gdf.geometry:
        coords = list(geom.coords)
        if len(coords) >= 2:
            msp.add_lwpolyline(
                coords, dxfattribs={"layer": f"ORIGINAL_{layer}"}
            )

    for geom in polygon_gdf.geometry:
        if geom.geom_type == "Polygon":
            msp.add_lwpolyline(
                list(geom.exterior.coords),
                close=True,
                dxfattribs={"layer": "DERIVED_Grenze"},
            )
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                msp.add_lwpolyline(
                    list(part.exterior.coords),
                    close=True,
                    dxfattribs={"layer": "DERIVED_Grenze"},
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)
    print(f"  Written: {output_path}")
    return True

"""dxf_extract — Extract DXF layers to GeoPackage files.

Reads one or more layers from a DXF file and writes each layer/geometry-type
combination to a separate GeoPackage.  Optionally strips the UTM zone prefix
from coordinates (common in CAD exports).

Use when: You need to get geometry out of a DXF into GIS-ready format for
further analysis or map production.

Example workflow.yaml::

    - name: Extract DXF layers
      template: dxf_extract
      params:
        dxf: Grundlagen/lageplan.dxf
        layers:
          - Baufeld
          - Wege
        crs: "EPSG:25833"
        strip_zone: true
      output: output/dxf_layers/
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "dxf_extract",
    description="Extract DXF layers to GeoPackage files",
    params=["dxf", "layers", "crs", "strip_zone"],
)
def dxf_extract(params: dict, project_dir: Path, output_path: Path) -> bool:
    """Extract geometry from DXF layers and save as GeoPackage.

    Params:
        dxf: Path to DXF file (relative to project root).
        layers: List of DXF layer names to extract.
        crs: Coordinate reference system (e.g. ``"EPSG:25833"``).
        strip_zone (optional): If ``true``, strip UTM zone prefix (32/33)
            from X coordinates.  Default ``false``.
    """
    from gis_utils import extract_dxf_layers, strip_utm_zone_prefix

    dxf_path = project_dir / params["dxf"]
    layers = params["layers"]
    crs = params["crs"]
    strip_zone = params.get("strip_zone", False)

    result = extract_dxf_layers(dxf_path, crs=crs, layers=layers)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for layer_name, geom_dict in result.items():
        for geom_type, gdf in geom_dict.items():
            if gdf.empty:
                continue
            if strip_zone:
                gdf = strip_utm_zone_prefix(gdf)
            out = output_path.parent / f"{layer_name}_{geom_type}.gpkg"
            gdf.to_file(out, driver="GPKG")
            print(f"  {layer_name}/{geom_type}: {len(gdf)} features -> {out}")
            written += 1

    return written > 0

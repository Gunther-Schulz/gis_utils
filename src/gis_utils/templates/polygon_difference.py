"""polygon_difference — input minus overlay (geometry difference).

Trivial workflow primitive: read two polygon layers, subtract overlay
from input, write the result. Equivalent to QGIS Processing's
``native:difference`` but pure-Python (no QGIS dependency).

Example workflow.yaml::

    - name: Build Reptilienschutzzaun
      template: polygon_difference
      params:
        input: Geodaten/Flurstück for Reptilienschutzzaun.shp
        overlay: Geodaten/Remove from Reptilienschutzzaun.gpkg
        crs: "EPSG:25833"
      output: Geodaten/Reptilienschutzzaun.gpkg

The output retains the **input**'s attributes (the overlay is treated as
a subtractive mask, not joined). Multi-feature inputs are supported:
each input geometry has the unioned overlay subtracted independently.
Empty results are dropped; remaining MultiPolygons are exploded to
single Polygons (per gis_utils output convention).
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "polygon_difference",
    description="Geometry difference: input layer minus overlay layer (pure-Python).",
    params=["input", "overlay", "crs", "input_layer", "overlay_layer", "output_layer"],
)
def polygon_difference(
    params: dict, project_dir: Path, output_path: Path | None
) -> bool:
    import geopandas as gpd

    if output_path is None:
        raise ValueError("polygon_difference requires an 'output:' path")

    input_path = (project_dir / params["input"]).resolve()
    overlay_path = (project_dir / params["overlay"]).resolve()
    crs = params.get("crs", "EPSG:25833")
    input_layer = params.get("input_layer")
    overlay_layer = params.get("overlay_layer")
    output_layer = params.get("output_layer", output_path.stem)

    inp = gpd.read_file(input_path, layer=input_layer) if input_layer else gpd.read_file(input_path)
    ovl = gpd.read_file(overlay_path, layer=overlay_layer) if overlay_layer else gpd.read_file(overlay_path)
    inp = inp.to_crs(crs)
    ovl = ovl.to_crs(crs)

    overlay_geom = ovl.geometry.union_all()
    inp["geometry"] = inp.geometry.difference(overlay_geom)
    inp = inp[~inp.geometry.is_empty]
    inp = inp.explode(index_parts=False).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gpkg":
        inp.to_file(output_path, driver="GPKG", layer=output_layer)
    else:
        inp.to_file(output_path)
    print(
        f"[polygon_difference] {output_path}: {len(inp)} feature(s), "
        f"area {inp.geometry.area.sum():.1f} m²"
    )
    return True

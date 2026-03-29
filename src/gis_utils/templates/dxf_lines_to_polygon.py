"""dxf_lines_to_polygon — Convert DXF polylines to a closed polygon boundary.

Extracts LineString geometry from a DXF layer and converts it into a single
closed polygon using the extend-and-polygonize approach: each line is extended
from both endpoints, the extended lines are noded at intersections, and the
resulting cells are unioned into the outer boundary polygon.

Use when: A DXF has polylines that form a boundary but don't quite connect
(small gaps at endpoints).  This is common with Materialwechsel, Baufeld, or
other boundary-type layers in Entwurfsplanungen.

If the result area is too small, the user can close remaining large gaps
manually in AutoCAD and re-run — the template picks up the changes.

Example workflow.yaml::

    - name: Materialwechsel Grenze
      template: dxf_lines_to_polygon
      params:
        dxf: Grundlagen/entwurf.dxf
        layer: PL_LIN_Materialwechsel
        crs: "EPSG:25833"
        strip_zone: true
        extend: 10.0
        snap_tolerance: 1.0
      output: output/grenze.gpkg
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "dxf_lines_to_polygon",
    description="Convert DXF polylines to a closed polygon (extend + polygonize)",
    params=["dxf", "layer", "crs", "strip_zone", "extend", "snap_tolerance"],
)
def dxf_lines_to_polygon(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
    """Extract lines from DXF and convert to a closed polygon.

    Params:
        dxf: Path to DXF file (relative to project root).
        layer: DXF layer name containing the polylines.
        crs: Coordinate reference system (e.g. ``"EPSG:25833"``).
        strip_zone (optional): Strip UTM zone prefix.  Default ``false``.
        extend (optional): Distance in metres to extend each line from both
            endpoints.  Default ``0`` (no extension).
        snap_tolerance (optional): Snap endpoints within this distance before
            extending.  Default ``0`` (no snapping).
        mode (optional): ``"outer"`` (default) returns exterior ring only.
            ``"all"`` returns union of all polygonized cells.
    """
    import geopandas as gpd

    from gis_utils import extract_dxf_layers, lines_to_polygon, strip_utm_zone_prefix

    dxf_path = project_dir / params["dxf"]
    layer = params["layer"]
    crs = params["crs"]
    strip_zone = params.get("strip_zone", False)
    extend = params.get("extend", 0)
    snap_tolerance = params.get("snap_tolerance", 0)
    mode = params.get("mode", "outer")

    layers = extract_dxf_layers(dxf_path, crs=crs, layers=[layer])
    gdf = layers[layer]["LineString"]

    if strip_zone:
        gdf = strip_utm_zone_prefix(gdf)

    polygon = lines_to_polygon(
        list(gdf.geometry),
        extend=extend,
        snap_tolerance=snap_tolerance,
        mode=mode,
    )

    print(
        f"  Polygon: {polygon.area:.1f} m² "
        f"({polygon.bounds[2]-polygon.bounds[0]:.0f}m x "
        f"{polygon.bounds[3]-polygon.bounds[1]:.0f}m)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"geometry": [polygon]}, crs=crs).to_file(
        output_path, driver="GPKG"
    )
    return True

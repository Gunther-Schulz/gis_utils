"""Built-in workflow templates for common GIS processing patterns.

Templates are Python functions that compose gis_utils library functions into
reusable processing steps.  They are invoked by the workflow runner via
``template:`` steps in workflow.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, Callable] = {}


def _register(name: str, *, description: str = "", params: list[str] | None = None):
    """Decorator to register a template handler."""

    def decorator(fn: Callable) -> Callable:
        fn._template_name = name
        fn._template_description = description
        fn._template_params = params or []
        _TEMPLATES[name] = fn
        return fn

    return decorator


def get_template(name: str) -> Callable:
    """Look up a template by name.

    Raises:
        KeyError: If *name* is not a registered template.
    """
    if name not in _TEMPLATES:
        raise KeyError(
            f"Unknown template '{name}'. "
            f"Available: {', '.join(sorted(_TEMPLATES))}"
        )
    return _TEMPLATES[name]


def list_templates() -> list[dict[str, Any]]:
    """Return metadata about all registered templates."""
    result = []
    for name, fn in sorted(_TEMPLATES.items()):
        result.append(
            {
                "name": name,
                "description": getattr(fn, "_template_description", ""),
                "params": getattr(fn, "_template_params", []),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Template implementations
# ---------------------------------------------------------------------------


@_register(
    "dxf_extract",
    description="Extract DXF layers to GeoPackage files",
    params=["dxf", "layers", "crs", "strip_zone"],
)
def _dxf_extract(params: dict, project_dir: Path, output_path: Path) -> bool:
    import geopandas as gpd

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


@_register(
    "dxf_lines_to_polygon",
    description="Convert DXF polylines to a closed polygon (extend + polygonize)",
    params=["dxf", "layer", "crs", "strip_zone", "extend", "snap_tolerance"],
)
def _dxf_lines_to_polygon(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
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


@_register(
    "verification_dxf",
    description="Write original DXF lines + derived polygon to DXF for visual QA",
    params=["dxf", "layer", "crs", "strip_zone", "polygon"],
)
def _verification_dxf(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
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

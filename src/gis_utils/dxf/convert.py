"""
Convert between shapefile and DXF formats.

Higher-level conversion utilities built on top of the lower-level
document creation and Map OD attachment modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf
import geopandas as gpd

from gis_utils.dxf.document import ensure_layer, new_dxf_document
from gis_utils.dxf.map_od import (
    attach_od_to_entity,
    encode_od_1004,
    get_table_handle_by_name,
)

# Approximate meters to degrees (WGS84, mid-latitudes)
_M_PER_DEG = 111320.0


def _geom_to_dxf(msp, geom, layer: str, dxfattribs: dict | None = None):
    """Add a Shapely geometry to DXF modelspace."""
    attribs = dict(dxfattribs or {})
    attribs.setdefault("layer", layer)
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Point":
        msp.add_point((geom.x, geom.y), dxfattribs=attribs)
    elif geom.geom_type in ("LineString", "LinearRing"):
        pts = list(geom.coords)
        if len(pts) >= 2:
            msp.add_lwpolyline([(x, y) for x, y in pts], format="xy", dxfattribs=attribs)
    elif geom.geom_type == "Polygon":
        pts = list(geom.exterior.coords)
        if len(pts) >= 2:
            msp.add_lwpolyline([(x, y) for x, y in pts], format="xy", close=True, dxfattribs=attribs)
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            _geom_to_dxf(msp, g, layer, dxfattribs)


def shapefile_to_dxf(
    shp_path: str | Path,
    dxf_path: str | Path,
    *,
    template_dxf_path: str | Path | None = None,
    layer: str = "0",
    crs: str | None = None,
    od_table_name: str | None = None,
    od_schema: list[str] | None = None,
    od_value_columns: list[str] | None = None,
    circle_radius_m: float = 0.0,
    mtext_content_fn: Any | None = None,
    mtext_height: float = 0.5,
) -> Path:
    """
    Convert a shapefile to DXF, optionally with Map Object Data and labels.

    For OD attachment, a template DXF from MAPIMPORT is required (it contains
    the OD table definitions that AutoCAD Map needs).

    Args:
        shp_path: Input shapefile path.
        dxf_path: Output DXF path.
        template_dxf_path: DXF template with OD table definitions (for OD attachment).
            If None, creates a new DXF document.
        layer: DXF layer name for all entities.
        crs: Override CRS for the shapefile. None = use shapefile's CRS.
        od_table_name: OD table name in the template (required for OD attachment).
        od_schema: OD field types (e.g. ["long", "string", "string"]).
        od_value_columns: Column names from shapefile to include in OD.
        circle_radius_m: If > 0, add circles at point locations with this radius (meters).
        mtext_content_fn: Optional callable(row, feat_id) -> str for MTEXT labels.
            Uses ``\\P`` for newlines in DXF. None = no labels.
        mtext_height: Character height for MTEXT labels.

    Returns:
        Path to the written DXF file.
    """
    shp_path = Path(shp_path)
    dxf_path = Path(dxf_path)

    # Load or create DXF document
    if template_dxf_path:
        template_dxf_path = Path(template_dxf_path)
        if not template_dxf_path.exists():
            raise FileNotFoundError(f"Template DXF not found: {template_dxf_path}")
        doc = ezdxf.readfile(str(template_dxf_path))
        # Clear existing entities
        msp = doc.modelspace()
        for entity in list(msp):
            entity.destroy()
    else:
        doc = new_dxf_document()

    ensure_layer(doc, layer)
    msp = doc.modelspace()

    # Resolve OD table handle
    table_handle = None
    if od_table_name and od_schema and od_value_columns:
        table_handle = get_table_handle_by_name(doc, od_table_name)
        if not table_handle:
            raise ValueError(f"OD table {od_table_name!r} not found in template")

    # Read shapefile
    gdf = gpd.read_file(shp_path)
    if crs:
        gdf = gdf.to_crs(crs)

    # Determine circle radius in CRS units
    circle_r = 0.0
    if circle_radius_m > 0:
        if gdf.crs and gdf.crs.is_geographic:
            circle_r = circle_radius_m / _M_PER_DEG
        else:
            circle_r = circle_radius_m

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        feat_id = idx + 1

        # Add geometry
        if geom.geom_type == "Point" and circle_r > 0:
            entity = msp.add_circle(
                (geom.x, geom.y), circle_r, dxfattribs={"layer": layer},
            )
        elif geom.geom_type == "Point":
            entity = msp.add_point((geom.x, geom.y), dxfattribs={"layer": layer})
        else:
            _geom_to_dxf(msp, geom, layer)
            entity = None  # OD only on point/circle entities for now

        # Attach OD
        if entity and table_handle and od_schema and od_value_columns:
            values = [feat_id]
            for col in od_value_columns:
                values.append(str(row.get(col) or "").strip())
            # Pad schema if needed (trailing longs for padding/ID)
            while len(values) < len(od_schema):
                values.append(row.get("ID", feat_id))
            binary = encode_od_1004(od_schema, values[:len(od_schema)])
            attach_od_to_entity(doc, entity, table_handle, feat_id, binary)

        # Add MTEXT label
        if entity and mtext_content_fn and geom.geom_type == "Point":
            content = mtext_content_fn(row, feat_id)
            offset = circle_r if circle_r > 0 else 1.0
            msp.add_mtext(
                content,
                dxfattribs={
                    "layer": layer,
                    "insert": (geom.x + offset * 0.5, geom.y + offset * 1.2),
                    "char_height": mtext_height if mtext_height > 0 else offset * 0.4,
                    "attachment_point": 1,
                },
            )

    dxf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(dxf_path))
    return dxf_path

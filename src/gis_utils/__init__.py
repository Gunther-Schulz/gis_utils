"""GIS utilities: DXF tools, geometry operations, reporting, WMS vectorization."""

from gis_utils.md_table import markdown_table
from gis_utils.dxf.document import new_dxf_document, ensure_layer
from gis_utils.dxf.extract import (
    extract_dxf_circles,
    extract_dxf_layers,
    interpolate_bulge_arc,
    lwpolyline_to_coords,
    save_layers_as_shapefiles,
)
from gis_utils.dxf.map_od import (
    OD_EXTENSION_DICT_KEY,
    attach_od_to_entity,
    encode_od_1004,
    get_table_handle_by_name,
)
from gis_utils.geometry import (
    find_column,
    load_and_union,
    make_valid_gdf,
    remove_inner_rings,
    subtract_geometries,
    subtract_smaller_overlaps,
)
from gis_utils.reporting import (
    area_by_category,
    area_report,
    intersection_areas,
)

__all__ = [
    # Markdown tables
    "markdown_table",
    # DXF document creation
    "new_dxf_document",
    "ensure_layer",
    # DXF extraction
    "extract_dxf_layers",
    "extract_dxf_circles",
    "interpolate_bulge_arc",
    "lwpolyline_to_coords",
    "save_layers_as_shapefiles",
    # DXF Map Object Data
    "OD_EXTENSION_DICT_KEY",
    "attach_od_to_entity",
    "encode_od_1004",
    "get_table_handle_by_name",
    # Geometry utilities
    "remove_inner_rings",
    "make_valid_gdf",
    "subtract_geometries",
    "subtract_smaller_overlaps",
    "load_and_union",
    "find_column",
    # Reporting
    "area_report",
    "area_by_category",
    "intersection_areas",
]

"""DXF utilities: document creation, geometry extraction, conversion, and Map Object Data."""

from gis_utils.dxf.convert import shapefile_to_dxf
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

__all__ = [
    "shapefile_to_dxf",
    "new_dxf_document",
    "ensure_layer",
    "extract_dxf_layers",
    "extract_dxf_circles",
    "interpolate_bulge_arc",
    "lwpolyline_to_coords",
    "save_layers_as_shapefiles",
    "OD_EXTENSION_DICT_KEY",
    "attach_od_to_entity",
    "encode_od_1004",
    "get_table_handle_by_name",
]

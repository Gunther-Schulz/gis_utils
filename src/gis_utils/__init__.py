"""GIS utilities: markdown tables and DXF helpers."""

from gis_utils.md_table import markdown_table
from gis_utils.dxf import new_dxf_document, ensure_layer
from gis_utils.map_od import (
    OD_EXTENSION_DICT_KEY,
    attach_od_to_entity,
    encode_od_1004,
    get_table_handle_by_name,
)

__all__ = [
    "markdown_table",
    "new_dxf_document",
    "ensure_layer",
    "encode_od_1004",
    "attach_od_to_entity",
    "get_table_handle_by_name",
    "OD_EXTENSION_DICT_KEY",
]

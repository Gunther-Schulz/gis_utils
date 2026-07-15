"""
CAD emitter: styled one-way export of GeoPackage/vector layers to DXF.

Public API:

* :func:`export_layers` + :class:`LayerSpec` — the emit entry point.
* :func:`load_styles` / :class:`Style` — the strict ``cad_styles.yaml`` schema.
* :func:`resolve_color` — ACI/RGB colour resolution from the packaged table.

New input contract (Phase-4 R1): GeoPackage sources + a strict style map in,
DXF out; no ``project_settings`` coupling, no sync. Legend, viewport, blocks,
operations, and manifest emission are out of scope for this cut.
"""

from gis_utils.cad.colors import ColorError, normalize_transparency, resolve_color
from gis_utils.cad.emit import (
    CAD_APP_ID,
    ExportError,
    LayerResult,
    LayerSpec,
    export_layers,
)
from gis_utils.cad.styles import (
    SCHEMA_VERSION,
    EntityStyle,
    HatchStyle,
    LayerStyle,
    Style,
    StyleError,
    TextStyle,
    load_styles,
    parse_styles,
)

__all__ = [
    "export_layers",
    "LayerSpec",
    "LayerResult",
    "ExportError",
    "CAD_APP_ID",
    "load_styles",
    "parse_styles",
    "Style",
    "LayerStyle",
    "HatchStyle",
    "TextStyle",
    "EntityStyle",
    "StyleError",
    "SCHEMA_VERSION",
    "resolve_color",
    "normalize_transparency",
    "ColorError",
]

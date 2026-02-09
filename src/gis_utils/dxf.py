"""
Helpers to create DXF documents with correct header, styles, and linetypes
so output works in CAD/GIS software.
"""

from typing import Any

import ezdxf


def new_dxf_document(version: str = "R2010") -> ezdxf.document.Drawing:
    """
    Create a new DXF document with proper drawing properties for CAD/GIS compatibility.

    Sets header vars ($MEASUREMENT, $INSUNITS, $LUNITS, etc.), default text styles
    (Standard, Arial), and common linetypes (CONTINUOUS, DASHED, DOTTED).

    Args:
        version: DXF version, e.g. "R2010" (default).

    Returns:
        New ezdxf document. Use doc.modelspace() to add entities, then doc.saveas(path).
    """
    doc = ezdxf.new(version)

    # Drawing properties so CAD/GIS software reads the file correctly
    doc.header["$MEASUREMENT"] = 1
    doc.header["$INSUNITS"] = 6
    doc.header["$LUNITS"] = 2
    doc.header["$LUPREC"] = 4
    doc.header["$AUPREC"] = 4
    doc.header["$ACADVER"] = "AC1024"
    doc.header["$DWGCODEPAGE"] = "ANSI_1252"

    # Default text styles
    for style_name, font in [("Standard", "Arial"), ("Arial", "Arial")]:
        if style_name not in doc.styles:
            try:
                style = doc.styles.new(style_name)
                style.dxf.font = font
                style.dxf.height = 0.0
                style.dxf.width = 1.0
            except Exception:
                pass

    # Common linetypes
    for lt, pattern in [
        ("CONTINUOUS", []),
        ("DASHED", [0.5, -0.25]),
        ("DOTTED", [0.0, -0.25]),
    ]:
        if lt not in doc.linetypes:
            try:
                doc.linetypes.add(lt, pattern=pattern)
            except Exception:
                pass

    return doc


def ensure_layer(
    doc: ezdxf.document.Drawing,
    name: str,
    color: int = 7,
    linetype: str = "CONTINUOUS",
) -> None:
    """
    Ensure a layer exists in the DXF document. Creates it if missing.

    Args:
        doc: DXF document (e.g. from new_dxf_document()).
        name: Layer name.
        color: ACI color 1–255 (7 = white/black by background).
        linetype: Linetype name (must exist in doc.linetypes).
    """
    if name in doc.layers:
        return
    try:
        doc.layers.add(name=name, color=color, linetype=linetype)
    except Exception:
        try:
            doc.layers.add(name=name)
        except Exception:
            pass

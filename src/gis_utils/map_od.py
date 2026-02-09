"""
Generic helper to attach AutoCAD Map Object Data (OD) to DXF entities using ezdxf.

Works with any entity type (POINT, CIRCLE, LWPOLYLINE, etc.), any OD table,
and any attribute schema. Not tied to a specific project, table name, or field
names — reusable in any project that needs Map-compatible OD in DXF.

Requirements:
- Document must already contain the OD table definition (IRD_DSC_DICT and
  IRD_DSC_RECORDs). Typically load a DXF created by MAPIMPORT as template,
  or create the schema separately. You need the table's IRD_DSC_RECORD handle
  (table_handle) to attach records.

Usage:
    import ezdxf
    from gis_utils import attach_od_to_entity, encode_od_1004, get_table_handle_by_name

    doc = ezdxf.readfile("template_with_od_tables.dxf")
    msp = doc.modelspace()
    point = msp.add_point((100, 200))

    # Schema: list of "long" (uint16 LE) or "string" (UTF-16 LE null-term)
    schema = ["long", "long", "string", "string", "string", "long"]
    values = [1, feat_id, "Name", "Tier", "Anzahl", id_val]
    binary_1004 = encode_od_1004(schema, values)

    table_handle = "AE"  # from your template's IRD_DSC_RECORD for the OD table
    attach_od_to_entity(doc, point, table_handle, record_index=1, binary_1004=binary_1004)
"""

from __future__ import annotations

import struct
from typing import Any, List, Optional

from ezdxf.lldxf.extendedtags import ExtendedTags
from ezdxf.lldxf.tags import Tags
from ezdxf.lldxf.types import DXFTag

# Extension dict key used by Map for the OD record (can override if needed)
OD_EXTENSION_DICT_KEY = "*A1"

# Constant stored in IRD_OBJ_RECORD 1070 (from reverse‑engineered Map export)
IRD_BASE_RECORD_1070 = 516


def encode_od_1004(
    schema: List[str],
    values: List[Any],
    *,
    leading_uint16: Optional[int] = 1,
) -> bytes:
    """
    Encode attribute values into the binary format used in Map OD 1004.

    Schema and values are generic: no hardcoded field names. Any project
    can define its own column order and types.

    Args:
        schema: List of type names in column order. Supported:
            - "long" → uint16 little-endian (2 bytes)
            - "string" → UTF-16 LE null-terminated string
        values: List of values in the same order as schema. Types must match:
            - "long": int (0–65535)
            - "string": str (any length; nulls/newlines normalized to space)
        leading_uint16: If not None, prepend this uint16 (e.g. 1) before the
            first schema field. Set to None to omit. Some Map tables use it.

    Returns:
        Raw bytes for group code 1004 (to pass to attach_od_to_entity).

    Example:
        schema = ["long", "string", "string", "long"]
        values = [1, "Name", "Comment", 42]
        binary = encode_od_1004(schema, values)
    """
    if len(schema) != len(values):
        raise ValueError("schema and values must have the same length")

    chunks: List[bytes] = []

    if leading_uint16 is not None:
        chunks.append(struct.pack("<H", leading_uint16 & 0xFFFF))

    for typ, val in zip(schema, values):
        if typ == "long":
            chunks.append(struct.pack("<H", int(val) & 0xFFFF))
        elif typ == "string":
            s = "" if val is None else str(val)
            s = s.replace("\r", " ").replace("\n", " ").strip()
            chunks.append(s.encode("utf-16-le") + b"\x00\x00")
        else:
            raise ValueError(f"Unknown schema type: {typ!r}")

    return b"".join(chunks)


def _build_ird_obj_record_tags(
    ird_handle: str,
    xdict_handle: str,
    table_handle: str,
    record_index: int,
    binary_1004: bytes,
) -> ExtendedTags:
    """Build ExtendedTags for one IRD_OBJ_RECORD (Map OD record)."""
    # 1004 must be hex string in ASCII DXF for round-trip; bytes would break reading
    value_1004 = binary_1004.hex() if isinstance(binary_1004, bytes) else binary_1004
    base = Tags.from_tuples([
        (0, "IRD_OBJ_RECORD"),
        (5, ird_handle),
    ])
    cird_base = Tags.from_tuples([
        (100, "CIrdBaseRecord"),
        (1070, IRD_BASE_RECORD_1070),
        (1071, record_index),
        (1004, value_1004),
    ])
    cird_obj = Tags.from_tuples([
        (100, "CIrdObjRecord"),
        (330, table_handle),
    ])
    extended = ExtendedTags()
    extended.subclasses.append(base)
    extended.subclasses.append(cird_base)
    extended.subclasses.append(cird_obj)
    return extended


def attach_od_to_entity(
    doc: "ezdxf.document.Drawing",
    entity: "ezdxf.entities.DXFEntity",
    table_handle: str,
    record_index: int,
    binary_1004: bytes,
    *,
    od_key: str = OD_EXTENSION_DICT_KEY,
) -> None:
    """
    Attach Map Object Data to any DXF entity (POINT, CIRCLE, LWPOLYLINE, etc.).

    Creates the entity's extension dictionary if needed, adds an IRD_OBJ_RECORD
    with the given 1004 payload, and links it so Map shows it as OD.

    Args:
        doc: The ezdxf Drawing (must have OBJECTS section and OD table in template).
        entity: Any graphic entity (POINT, LINE, CIRCLE, LWPOLYLINE, etc.).
        table_handle: Handle of the IRD_DSC_RECORD that defines the OD table
            (e.g. from a MAPIMPORT template). Required; no default.
        record_index: Value for 1071 (record index, typically 1-based).
        binary_1004: Raw bytes for the OD payload (group 1004). Use
            encode_od_1004(schema, values) for the standard Map format.
        od_key: Key in the extension dictionary (default "*A1" for Map).
    """
    if not entity.is_alive or entity.doc is not doc:
        raise ValueError("entity must be alive and belong to doc")

    # Get or create extension dictionary on the entity
    if entity.has_extension_dict:
        xdict = entity.get_extension_dict()
    else:
        xdict = entity.new_extension_dict()

    xdict_handle = xdict.handle

    # Create IRD_OBJ_RECORD in OBJECTS
    ird = doc.objects.new_entity("IRD_OBJ_RECORD", {})
    ird_handle = ird.dxf.handle

    # Build and store tags (base + CIrdBaseRecord + CIrdObjRecord)
    extended = _build_ird_obj_record_tags(
        ird_handle,
        xdict_handle,
        table_handle,
        record_index,
        binary_1004,
    )
    ird.store_tags(extended)

    # Link IRD to extension dict (owner and reactor)
    ird.dxf.owner = xdict_handle
    ird.set_reactors([xdict_handle])

    # Attach IRD to extension dict so Map finds it
    xdict[od_key] = ird


def get_table_handle_by_name(
    doc: "ezdxf.document.Drawing",
    table_name: str,
) -> Optional[str]:
    """
    Try to find the handle of an IRD_DSC_RECORD (OD table) by table name.

    Best-effort: searches OBJECTS for IRD_DSC_RECORD entities and decodes
    their 1004 payload for a matching table name. Returns None if not found
    or format is unexpected. Prefer using a known handle from your template.

    Args:
        doc: The ezdxf Drawing (e.g. loaded from a MAPIMPORT template).
        table_name: Exact table name (e.g. "Ersatzquartiere_QaM").

    Returns:
        Handle of the IRD_DSC_RECORD for that table, or None.
    """
    try:
        for obj in doc.objects:
            dt = getattr(obj, "dxftype", None)
            if not callable(dt) or dt() != "IRD_DSC_RECORD":
                continue
            xtags = getattr(obj, "xtags", None)
            if xtags is None:
                continue
            subclasses = getattr(xtags, "subclasses", [])
            for subclass in subclasses:
                for tag in subclass:
                    if getattr(tag, "code", None) != 1004:
                        continue
                    raw = getattr(tag, "value", None)
                    if isinstance(raw, bytes):
                        try:
                            s = raw.decode("utf-16-le", errors="replace")
                            if table_name in s:
                                return getattr(obj.dxf, "handle", None)
                        except Exception:
                            pass
                    elif isinstance(raw, str) and len(raw) % 2 == 0:
                        try:
                            b = bytes.fromhex(raw)
                            s = b.decode("utf-16-le", errors="replace")
                            if table_name in s:
                                return getattr(obj.dxf, "handle", None)
                        except Exception:
                            pass
    except Exception:
        pass
    return None

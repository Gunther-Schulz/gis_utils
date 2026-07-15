"""
Thin block-reference and MTEXT insertion over ezdxf.

New rewrite (Phase-4 R1) of the core of Python-ACAD-Tools'
``block_insert_manager.py`` / ``text_insert_manager.py``. The reference
managers were built on the bidirectional ``UnifiedSyncProcessor`` (discovery,
pull, hash-based sync, YAML round-trip). None of that belongs in a one-way
emitter, so this keeps only the two core placement calls:

* :func:`insert_block` — reference a block **already defined** in the document
  (typically supplied by a template DXF); this module never defines blocks.
* :func:`add_text` — place an MTEXT with an optional text style.

Both tag the created entity with the emitter provenance so an idempotent
re-export clears them.
"""

from __future__ import annotations

from gis_utils.cad.colors import resolve_color
from gis_utils.cad.emit import _attach_provenance, _ensure_text_style
from gis_utils.cad.styles import ATTACHMENT_CODES, TextStyle


class AnnotateError(ValueError):
    """Raised for an invalid block/text insertion (unknown block, bad space)."""


def _space(doc, space: str):
    if space == "model":
        return doc.modelspace()
    if space == "paper":
        return doc.paperspace()
    raise AnnotateError(f"space must be 'model' or 'paper', got {space!r}")


def insert_block(
    doc,
    name: str,
    position: tuple[float, float],
    *,
    layer: str = "0",
    scale: float = 1.0,
    rotation: float = 0.0,
    space: str = "model",
):
    """
    Insert a reference to a block already defined in *doc*.

    Args:
        doc: ezdxf document. The block *name* must exist in ``doc.blocks``
            (e.g. brought in via a template DXF) — this module does not define
            blocks, so an unknown name is a hard error, never a silent no-op.
        name: Block definition name to reference.
        position: Insertion point ``(x, y)``.
        layer: Target layer (created if missing).
        scale: Uniform x/y/z scale.
        rotation: Rotation in degrees.
        space: ``"model"`` or ``"paper"``.

    Returns:
        The created INSERT entity.

    Raises:
        AnnotateError: if the block is not defined in the document.
    """
    if not name or name not in doc.blocks:
        available = [b.name for b in doc.blocks][:10]
        raise AnnotateError(
            f"Block {name!r} not defined in document; provide it via a template. "
            f"Known (first 10): {available}"
        )
    if layer not in doc.layers:
        doc.layers.add(layer)
    sp = _space(doc, space)
    ref = sp.add_blockref(
        name, position,
        dxfattribs={"layer": layer, "xscale": scale, "yscale": scale,
                    "zscale": scale, "rotation": rotation},
    )
    _attach_provenance(ref, doc)
    return ref


def add_text(
    doc,
    text: str,
    position: tuple[float, float],
    *,
    style: TextStyle | None = None,
    layer: str = "0",
    space: str = "model",
):
    """
    Add an MTEXT entity, optionally styled by a :class:`TextStyle`.

    Args:
        doc: ezdxf document.
        text: Text content (``\\n`` accepted as line breaks by MTEXT).
        position: Insertion point ``(x, y)``.
        style: Optional :class:`TextStyle` (font, height, colour, attachment,
            rotation, max width). ``None`` uses ezdxf defaults on the layer.
        layer: Target layer (created if missing).
        space: ``"model"`` or ``"paper"``.

    Returns:
        The created MTEXT entity.
    """
    if layer not in doc.layers:
        doc.layers.add(layer)
    sp = _space(doc, space)

    attribs = {"layer": layer, "insert": position}
    if style is not None:
        attribs["style"] = _ensure_text_style(doc, style)
        attribs["char_height"] = style.height
        if style.max_width:
            attribs["width"] = style.max_width

    mtext = sp.add_mtext(str(text), dxfattribs=attribs)

    if style is not None:
        if style.color is not None:
            color = resolve_color(style.color)
            if isinstance(color, tuple):
                mtext.rgb = color
            else:
                mtext.dxf.color = color
        if style.attachment is not None:
            mtext.dxf.attachment_point = ATTACHMENT_CODES[style.attachment]
        if style.rotation is not None:
            mtext.dxf.rotation = float(style.rotation)

    _attach_provenance(mtext, doc)
    return mtext

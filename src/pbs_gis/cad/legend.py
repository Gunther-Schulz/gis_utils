"""
Simple DXF legend: a flat column of swatch + label rows.

New, deliberately schlicht rewrite (Phase-4 R1) of Python-ACAD-Tools'
``legend_creator.py``. The reference carried nested groups, subtitles, presets,
block symbols and a mutable ``current_y`` engine bound to ``project_loader``.
This version keeps none of that state: the caller passes a list of
:class:`LegendEntry` and a :class:`LegendStyle`, and each entry becomes one row
(a coloured/hatched swatch box or a line sample, plus an MTEXT label). Entries
reuse the emitter's :class:`~pbs_gis.cad.styles.Style` so a legend swatch is
coloured and hatched exactly like the geometry it stands for.

All emitted entities carry the emitter provenance tag, so an idempotent
re-export (:func:`pbs_gis.cad.emit.export_layers`) clears the legend together
with the rest of the emitter's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pbs_gis.cad.colors import resolve_color
from pbs_gis.cad.emit import _add_hatch, _attach_provenance, _polygon_rings
from pbs_gis.cad.styles import Style

# Swatch kinds. ``area`` draws a filled/outlined box, ``line`` a horizontal
# line sample, ``empty`` a bare outline box (no fill, no line).
_KINDS = frozenset({"area", "line", "empty"})


@dataclass
class LegendEntry:
    """One legend row.

    Attributes:
        label: Text drawn to the right of the swatch.
        kind: ``"area"`` (box, hatched if the style carries a hatch),
            ``"line"`` (horizontal line sample) or ``"empty"`` (outline box).
        style: The emitter :class:`Style` whose layer/hatch colour drives the
            swatch. ``None`` draws a plain BYLAYER swatch.
    """

    label: str
    kind: str = "area"
    style: Style | None = None


@dataclass
class LegendStyle:
    """Geometry of the legend block (all lengths in drawing units)."""

    swatch_width: float = 8.0
    swatch_height: float = 4.0
    row_spacing: float = 2.0
    text_offset: float = 3.0
    text_height: float = 2.5
    text_font: str = "Standard"
    layer: str = "Legend"


@dataclass
class LegendResult:
    """Outcome of :func:`add_legend` (for verification/chaining)."""

    layer: str
    rows: int = 0
    hatches: int = 0
    end_y: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _swatch_color(style: Style | None):
    """ACI/RGB for a swatch outline: the style's layer colour if any."""
    if style is not None and style.layer is not None and style.layer.color is not None:
        return resolve_color(style.layer.color)
    return None


def _apply_color(entity, color) -> None:
    if color is None:
        return
    if isinstance(color, tuple):
        entity.rgb = color
    else:
        entity.dxf.color = color


def add_legend(
    doc,
    entries: list[LegendEntry],
    *,
    position: tuple[float, float],
    style: LegendStyle | None = None,
) -> LegendResult:
    """
    Draw a flat legend (one row per entry) into modelspace.

    Args:
        doc: An ezdxf document (typically the one returned/loaded by the
            emitter; the legend is added to modelspace).
        entries: Rows top-to-bottom.
        position: ``(x, y)`` of the top-left corner of the first swatch.
        style: Geometry/appearance; defaults to :class:`LegendStyle`.

    Returns:
        A :class:`LegendResult` with the layer name, row/hatch counts and the
        ``y`` below the last row.

    Raises:
        ValueError: on an unknown entry ``kind``.
    """
    style = style or LegendStyle()
    msp = doc.modelspace()
    if style.layer not in doc.layers:
        doc.layers.add(style.layer)

    if style.text_font not in doc.styles:
        try:
            ts = doc.styles.new(style.text_font)
            ts.dxf.font = style.text_font
        except Exception:
            pass

    result = LegendResult(layer=style.layer, end_y=position[1])
    x0, y_top = position
    cur_y = y_top
    x1 = x0 + style.swatch_width

    for entry in entries:
        if entry.kind not in _KINDS:
            raise ValueError(
                f"Unknown legend entry kind {entry.kind!r}; allowed: {sorted(_KINDS)}"
            )
        y0 = cur_y
        y1 = cur_y - style.swatch_height
        color = _swatch_color(entry.style)

        if entry.kind in ("area", "empty"):
            box = msp.add_lwpolyline(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                format="xy", close=True, dxfattribs={"layer": style.layer},
            )
            _apply_color(box, color)
            _attach_provenance(box, doc)

            if entry.kind == "area" and entry.style is not None and entry.style.hatch is not None:
                rings = _polygon_rings(
                    _box_polygon(x0, y0, x1, y1)
                )
                # Force the hatch onto the legend layer (ignore any layer_suffix
                # so the sample stays inside the legend block).
                hs = entry.style.hatch
                h = _add_hatch(msp, rings, _no_suffix(hs), style.layer, doc)
                if h is not None:
                    _attach_provenance(h, doc)
                    result.hatches += 1

        elif entry.kind == "line":
            mid = (y0 + y1) / 2
            line = msp.add_lwpolyline(
                [(x0, mid), (x1, mid)], format="xy",
                dxfattribs={"layer": style.layer},
            )
            _apply_color(line, color)
            _attach_provenance(line, doc)

        text_y = (y0 + y1) / 2
        mtext = msp.add_mtext(
            entry.label,
            dxfattribs={
                "layer": style.layer,
                "style": style.text_font,
                "char_height": style.text_height,
                "insert": (x1 + style.text_offset, text_y),
                "attachment_point": 4,  # MIDDLE_LEFT
            },
        )
        _attach_provenance(mtext, doc)

        result.rows += 1
        cur_y = y1 - style.row_spacing

    result.end_y = cur_y
    return result


def _box_polygon(x0, y0, x1, y1):
    from shapely.geometry import Polygon

    return Polygon([(x0, y1), (x1, y1), (x1, y0), (x0, y0)])


def _no_suffix(hatch_style):
    """A copy of *hatch_style* with ``layer_suffix`` cleared (frozen dataclass)."""
    from dataclasses import replace

    return replace(hatch_style, layer_suffix=None)

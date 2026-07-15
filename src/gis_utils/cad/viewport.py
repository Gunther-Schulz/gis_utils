"""
Paperspace viewport creation over ezdxf.

New rewrite (Phase-4 R1) of the core of Python-ACAD-Tools'
``viewport_manager.py``. That 668-line class was almost entirely
bidirectional-sync machinery (``UnifiedSyncProcessor`` discovery, pull, hash,
corruption checks, YAML round-trip). The genuinely reusable kernel is small and
well-defined: place a paperspace viewport with a window onto a modelspace
region, in strict 2D. That is what this module provides:

* :func:`add_viewport` — the low primitive (explicit paper rect + view window).
* :func:`add_viewport_for_bbox` — fit a modelspace bounding box into a paper
  rectangle, computing the view centre and height (and the resulting scale).

Everything the reference did beyond this — sync direction, discovery of
hand-made viewports, clip paths, frozen-layer diffing — is intentionally left
out and reported as a gap; it has no place in a one-way emitter.
"""

from __future__ import annotations

from dataclasses import dataclass

from gis_utils.cad.emit import _attach_provenance


@dataclass
class ViewportResult:
    """Outcome of a viewport insertion (for verification/reporting)."""

    viewport: object
    view_center: tuple[float, float]
    view_height: float
    scale: float  # model units per paper unit (view_height / paper_height)


def _ensure_vp_layer(doc, layer: str):
    if layer not in doc.layers:
        doc.layers.add(layer)
    # Viewport frames conventionally do not plot.
    doc.layers.get(layer).dxf.plot = 0


def _set_2d(viewport) -> None:
    """Force strict top-down 2D behaviour on a viewport."""
    viewport.dxf.status = 1
    viewport.dxf.render_mode = 0  # 2D optimised
    viewport.dxf.view_direction_vector = (0, 0, 1)


def add_viewport(
    doc,
    *,
    paper_center: tuple[float, float],
    paper_size: tuple[float, float],
    view_center: tuple[float, float],
    view_height: float,
    layer: str = "VIEWPORTS",
) -> ViewportResult:
    """
    Add a paperspace viewport with an explicit view window.

    Args:
        doc: ezdxf document.
        paper_center: ``(x, y)`` centre of the viewport rectangle on the sheet.
        paper_size: ``(width, height)`` of the viewport rectangle on the sheet.
        view_center: ``(x, y)`` modelspace point shown at the viewport centre.
        view_height: Modelspace height mapped onto the viewport height (sets
            the scale; the width follows from the paper aspect ratio).
        layer: Viewport-frame layer (created, set non-plotting).

    Returns:
        A :class:`ViewportResult`.
    """
    _ensure_vp_layer(doc, layer)
    psp = doc.paperspace()
    vp = psp.add_viewport(
        center=paper_center,
        size=paper_size,
        view_center_point=view_center,
        view_height=view_height,
    )
    vp.dxf.layer = layer
    _set_2d(vp)
    _attach_provenance(vp, doc)
    paper_h = paper_size[1]
    scale = (view_height / paper_h) if paper_h else 0.0
    return ViewportResult(vp, view_center, view_height, scale)


def add_viewport_for_bbox(
    doc,
    bbox: tuple[float, float, float, float],
    *,
    paper_center: tuple[float, float],
    paper_size: tuple[float, float],
    margin: float = 0.05,
    layer: str = "VIEWPORTS",
) -> ViewportResult:
    """
    Fit a modelspace bounding box into a paperspace viewport rectangle.

    The view is centred on the bbox centre and the view height is chosen so the
    whole bbox is visible at the paper rectangle's aspect ratio, plus a margin.

    Args:
        doc: ezdxf document.
        bbox: ``(minx, miny, maxx, maxy)`` in modelspace units.
        paper_center: Centre of the viewport rectangle on the sheet.
        paper_size: ``(width, height)`` of the viewport rectangle.
        margin: Fractional padding around the bbox (``0.05`` = 5% each side).
        layer: Viewport-frame layer.

    Returns:
        A :class:`ViewportResult` (its ``scale`` is model units per paper unit).

    Raises:
        ValueError: for a degenerate bbox or non-positive paper size.
    """
    minx, miny, maxx, maxy = bbox
    bw, bh = maxx - minx, maxy - miny
    pw, ph = paper_size
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate bbox {bbox!r} (zero width or height)")
    if pw <= 0 or ph <= 0:
        raise ValueError(f"paper_size must be positive, got {paper_size!r}")

    view_center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

    # view_width / view_height must equal the paper aspect. Pick the view height
    # that contains the bbox in both directions.
    aspect = pw / ph
    view_height = max(bh, bw / aspect)
    view_height *= (1.0 + 2.0 * margin)

    return add_viewport(
        doc,
        paper_center=paper_center,
        paper_size=paper_size,
        view_center=view_center,
        view_height=view_height,
        layer=layer,
    )

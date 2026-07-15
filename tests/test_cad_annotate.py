"""Unit tests for the CAD annotate layer: legend, block/text insertion, viewport."""

from __future__ import annotations

import ezdxf
import pytest

from gis_utils.cad import (
    AnnotateError,
    LegendEntry,
    LegendStyle,
    Style,
    add_legend,
    add_text,
    add_viewport,
    add_viewport_for_bbox,
    insert_block,
    parse_styles,
)
from gis_utils.cad.emit import CAD_APP_ID, _purge_emitted
from gis_utils.dxf.document import new_dxf_document


def _styles():
    return parse_styles({
        "schema_version": 1,
        "styles": {
            "area": {"layer": {"color": "green"},
                     "hatch": {"pattern": "SOLID", "color": "green"}},
            "line": {"layer": {"color": "red"}},
        },
    })


# --- legend ----------------------------------------------------------------

def test_add_legend_rows_and_hatch(tmp_path):
    doc = new_dxf_document()
    smap = _styles()
    entries = [
        LegendEntry("Baufeld", kind="area", style=smap["area"]),
        LegendEntry("Baugrenze", kind="line", style=smap["line"]),
        LegendEntry("Leer", kind="empty"),
    ]
    result = add_legend(doc, entries, position=(0, 100))
    assert result.rows == 3
    assert result.hatches == 1
    msp = doc.modelspace()
    assert "Legend" in doc.layers
    assert len(msp.query('MTEXT[layer=="Legend"]')) == 3
    assert len(msp.query('HATCH[layer=="Legend"]')) == 1
    # area + empty boxes = 2 closed polylines; line sample = 1 open polyline.
    assert len(msp.query('LWPOLYLINE[layer=="Legend"]')) == 3
    # end_y is below the starting y.
    assert result.end_y < 100


def test_legend_unknown_kind_raises():
    doc = new_dxf_document()
    with pytest.raises(ValueError):
        add_legend(doc, [LegendEntry("x", kind="bogus")], position=(0, 0))


def test_legend_entities_are_provenance_tagged():
    doc = new_dxf_document()
    add_legend(doc, [LegendEntry("A", kind="empty")], position=(0, 0))
    removed = _purge_emitted(doc)
    assert removed >= 2  # box + label
    assert len(doc.modelspace().query('*[layer=="Legend"]')) == 0


def test_legend_custom_style():
    doc = new_dxf_document()
    style = LegendStyle(swatch_width=20, layer="MyLegend")
    add_legend(doc, [LegendEntry("A", kind="empty")], position=(0, 0), style=style)
    assert "MyLegend" in doc.layers


# --- blocks / text ---------------------------------------------------------

def test_insert_block_from_definition():
    doc = new_dxf_document()
    block = doc.blocks.new(name="pv_symbol")
    block.add_circle((0, 0), radius=1)
    ref = insert_block(doc, "pv_symbol", (10, 20), layer="Symbols", scale=2.0, rotation=90)
    assert ref.dxf.name == "pv_symbol"
    assert ref.dxf.insert.x == 10 and ref.dxf.insert.y == 20
    assert ref.dxf.xscale == 2.0
    assert "Symbols" in doc.layers
    assert len(doc.modelspace().query('INSERT[layer=="Symbols"]')) == 1


def test_insert_unknown_block_raises():
    doc = new_dxf_document()
    with pytest.raises(AnnotateError):
        insert_block(doc, "not_defined", (0, 0))


def test_add_text_with_style():
    doc = new_dxf_document()
    smap = parse_styles({
        "schema_version": 1,
        "styles": {"t": {"text": {"height": 3.0, "color": "green",
                                  "attachment": "MIDDLE_CENTER", "rotation": 45}}},
    })
    mtext = add_text(doc, "Hallo", (5, 5), style=smap["t"].text, layer="Plantext")
    assert mtext.dxf.char_height == 3.0
    assert mtext.dxf.rotation == 45
    assert "Plantext" in doc.layers
    assert len(doc.modelspace().query('MTEXT[layer=="Plantext"]')) == 1


def test_add_text_in_paperspace():
    doc = new_dxf_document()
    add_text(doc, "P", (0, 0), space="paper")
    assert len(doc.paperspace().query("MTEXT")) == 1
    assert len(doc.modelspace().query("MTEXT")) == 0


def test_block_and_text_are_tagged():
    doc = new_dxf_document()
    doc.blocks.new(name="b").add_point((0, 0))
    insert_block(doc, "b", (0, 0))
    add_text(doc, "x", (1, 1))
    assert _purge_emitted(doc) == 2


# --- viewport --------------------------------------------------------------

def test_add_viewport_explicit():
    doc = new_dxf_document()
    result = add_viewport(
        doc,
        paper_center=(100, 100),
        paper_size=(180, 120),
        view_center=(500, 500),
        view_height=240,
    )
    vp = result.viewport
    assert vp.dxf.center.x == 100 and vp.dxf.center.y == 100
    assert vp.dxf.view_center_point.x == 500
    assert vp.dxf.view_height == 240
    assert result.scale == 2.0  # 240 / 120
    assert "VIEWPORTS" in doc.layers
    assert doc.layers.get("VIEWPORTS").dxf.plot == 0
    assert len(doc.paperspace().query("VIEWPORT")) >= 1


def test_add_viewport_for_bbox_fits_and_centers():
    doc = new_dxf_document()
    bbox = (0, 0, 100, 50)  # width 100, height 50
    result = add_viewport_for_bbox(
        doc, bbox, paper_center=(0, 0), paper_size=(200, 100), margin=0.0,
    )
    # Centered on bbox center.
    assert result.view_center == (50.0, 25.0)
    # Paper aspect 2:1 matches bbox aspect 2:1 → view_height == bbox height.
    assert result.view_height == pytest.approx(50.0)


def test_add_viewport_for_bbox_aspect_mismatch():
    doc = new_dxf_document()
    # Tall bbox into a wide paper: view must grow in height to fit width.
    bbox = (0, 0, 100, 10)
    result = add_viewport_for_bbox(
        doc, bbox, paper_center=(0, 0), paper_size=(100, 100), margin=0.0,
    )
    # aspect = 1; view_height = max(10, 100/1) = 100.
    assert result.view_height == pytest.approx(100.0)


def test_add_viewport_for_bbox_degenerate_raises():
    doc = new_dxf_document()
    with pytest.raises(ValueError):
        add_viewport_for_bbox(doc, (0, 0, 0, 10), paper_center=(0, 0), paper_size=(10, 10))

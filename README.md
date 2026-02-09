# gis_utils

GIS utilities: markdown tables with fixed-width columns and DXF helpers for CAD-compatible output.

## Install

```bash
pip install gis-utils
```

From GitHub (before publishing to PyPI):

```bash
pip install git+https://github.com/Gunther-Schulz/gis_utils.git
```

## Usage

### Markdown tables

Build markdown tables that align in raw view and render correctly:

```python
from gis_utils import markdown_table

table = markdown_table(
    headers=["Name", "Area", "Count"],
    rows=[
        ["Site A", 1234.5, 10],
        ["Site B", 567.8, 3],
    ],
    align=["l", "r", "r"],
)
print(table)
```

### DXF documents

Create a DXF document with correct header, text styles, and linetypes so it works in CAD/GIS:

```python
from gis_utils import new_dxf_document, ensure_layer

doc = new_dxf_document(version="R2010")
msp = doc.modelspace()

ensure_layer(doc, "MyLayer", color=7, linetype="CONTINUOUS")
msp.add_lwpolyline([(0, 0), (10, 0), (10, 10)], dxfattribs={"layer": "MyLayer"})

doc.saveas("output.dxf")
```

## License

MIT

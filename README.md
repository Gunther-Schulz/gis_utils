# gis_utils

GIS/CAD utility library and project workflow runner.

## Install

Editable install (recommended for development — edits take effect immediately):

```bash
conda activate gis
pip install -e ~/dev/Gunther-Schulz/gis_utils
```

## Starting a new project

```bash
# 1. Create project folder
mkdir "My Project"

# 2. Initialize (creates workflow.yaml, scripts/, CLAUDE.md)
gis-workflow init "My Project"

# 3. Edit CLAUDE.md — fill in the "Project notes" section with:
#    - CRS (e.g. EPSG:25832)
#    - Data sources and locations
#    - Any coordinate quirks or project-specific context

# 4. Start Claude session in the project folder — it reads CLAUDE.md
#    and knows how to use gis_utils, the workflow runner, and where
#    to put project-specific vs reusable code.
```

## Workflow runner

Each project has a `workflow.yaml` defining the execution pipeline.

```bash
# Run full workflow
gis-workflow run

# Preview execution plan
gis-workflow --dry-run

# Run single step + its dependencies
gis-workflow --step "Step Name"

# Initialize new project
gis-workflow init [project_dir]
```

Steps marked `run: once` are skipped if their outputs already exist.
Steps marked `run: always` execute every time.
Dependencies are resolved automatically (topological sort).

## Library API

All common functions importable from top level: `from gis_utils import ...`

### DXF
- `extract_dxf_layers()` — extract all geometry from DXF → `{layer: {geom_type: GeoDataFrame}}`
- `extract_dxf_circles()` — circle centers as Point GeoDataFrame with radius
- `save_layers_as_shapefiles()` — write extracted layers to organized shapefiles
- `new_dxf_document()` — new DXF with proper CAD headers
- `ensure_layer()` — create layer if missing
- `shapefile_to_dxf()` — convert SHP→DXF with optional Map OD and labels
- `attach_od_to_entity()` / `encode_od_1004()` — AutoCAD Map Object Data

### Geometry
- `remove_inner_rings()` — remove holes from polygons
- `make_valid_gdf()` — repair invalid geometries
- `subtract_geometries()` — set difference (base minus subtract)
- `subtract_smaller_overlaps()` — remove overlaps by area
- `morphological_filter()` — buffer-dissolve-buffer polygon cleanup
- `distance_to_nearest()` — min distance to reference features
- `points_with_buffers()` — create points + buffer union from coordinate data
- `load_and_union()` — load shapefile, union all geometries
- `find_column()` — find column by name variants

### Reporting
- `markdown_table()` — fixed-width markdown table (aligns in raw view)
- `area_report()` — full markdown area report with optional parcel intersection
- `area_by_category()` — intersection areas grouped by category
- `intersection_areas()` — intersection area per parcel

### Specialized (import from submodule)
- `from gis_utils.osm import download_osm_polygons` — OSM Overpass API
- `from gis_utils.wms import run` — WMS download + vectorization
- `from gis_utils.grass import main` — GRASS GIS centerline extraction

## Full API reference

See [CLAUDE.md](CLAUDE.md) for the complete API with signatures.

## License

MIT

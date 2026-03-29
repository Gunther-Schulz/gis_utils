# gis_utils

GIS/CAD utility library. Installed as editable in conda env `gis`:
```
pip install -e ~/dev/Gunther-Schulz/gis_utils
```

## Alpha stage — no backward compatibility

This library is in alpha. Do not add backward-compatibility shims, deprecated aliases, re-exports of renamed symbols, or any code whose sole purpose is keeping old callers working. When something changes, just change it. All callers are in our own projects and can be updated immediately.

## Output convention: always single Polygons, not MultiPolygons

When producing GeoDataFrame outputs, always explode MultiPolygons into individual Polygon features (`.explode(index_parts=False)`). This applies to all scripts and library functions that write GeoPackage/Shapefile outputs. MultiPolygons make styling, labeling, and area calculations unreliable in QGIS.

## CRITICAL: No dangerous defaults or silent fallbacks

**Be extremely careful with default parameter values and fallback patterns** (e.g. `x = x or SOME_DEFAULT`). Silent defaults can cause hard-to-detect data corruption. Rules:

- **CRS**: NEVER default to a specific EPSG code. Different projects use different zones (25832 vs 25833 etc.). A wrong CRS silently shifts geometries by hundreds of meters. Always require CRS explicitly.
- **URLs, layer names, file paths**: NEVER hardcode project-specific values as defaults. Require them as parameters or get them from recipes.
- **Any parameter where a wrong default produces valid-looking but incorrect output**: make it required, not optional with a default.
- **Safe defaults are OK**: things like `timeout=120`, `dissolve=True`, `simplify_tolerance=1.0` — where a wrong value causes obvious failures or minor quality differences, not silent corruption.
- **When in doubt**: require the parameter with no default. An explicit error is always better than silently wrong data.

## Geometry tasks: analyze data before coding

For geometry conversion/analysis tasks (lines to polygons, closing gaps, finding boundaries, extracting features, etc.), **always analyze the data before writing any solution code**:

1. **Inspect the data**: count features, types, lengths, orientations
2. **Measure relationships**: gaps between endpoints, connectivity, parallel vs crossing
3. **Ask the user** if the physical interpretation is unclear ("these look like two parallel boundary lines with cross-lines between them — is that right?")
4. **Think: how would a user solve this manually?** What steps would they take in AutoCAD, QGIS, or another GIS application? The manual workflow almost always reveals the simplest automated approach:
   - AutoCAD: extend, trim, close, join, hatch boundary, polyline edit
   - QGIS: buffer, dissolve, merge, polygonize, select by location, field calculator
   - The manual steps translate directly to shapely/geopandas operations
5. **Start with that naive approach FIRST.** Only escalate to computational geometry algorithms (concave hull, alpha shapes, graph traversal) if the simple approach actually fails on the data

Example: 13 disconnected polylines with 5–18 m gaps → a user in AutoCAD would just extend the lines and trim. Don't reach for concave hull or planar graph traversal. Just `extend_line` + `polygonize`. Done.

## Discovery / Catalog

Before reading source files, use the auto-generated catalog to discover what's available:

```python
from gis_utils import catalog
result = catalog()                          # everything
result = catalog(search="dxf")              # filtered
result = catalog(project_dir="/path/to/project")  # include project-local recipes
```

CLI: `gis-workflow catalog [--search TERM] [project_dir]`

Returns a dict with `version`, `functions` (grouped by module with signatures), `recipes` (with layers/tags), and `cli` commands.

### IMPORTANT: Keeping the catalog in sync

The catalog auto-discovers functions via `inspect` and recipes via `list_recipes()`, so most changes are picked up automatically. However, **you must update `catalog.py`** when:

- **Adding a new submodule** — add it to the `_MODULES` list in `catalog.py` so its functions are discovered
- **Adding a new CLI subcommand** — add it to the `_CLI_COMMANDS` list in `catalog.py`

Function signatures and docstrings are introspected automatically — no manual updates needed for those.

## API Quick Reference

All common functions importable from top level: `from gis_utils import ...`

### DXF Extraction
- `extract_dxf_layers(dxf_path, crs, *, layers, exclude_layers, circles_as_points, process_blocks)` — extract all geometry from DXF, returns `{layer: {geom_type: GeoDataFrame}}`
- `extract_dxf_circles(dxf_path, crs, *, layers)` — circle centers as Point GeoDataFrame with `radius` column
- `lwpolyline_to_coords(entity, arc_points)` — LWPOLYLINE/POLYLINE to coordinate list with bulge interpolation
- `interpolate_bulge_arc(start, end, bulge, num_points)` — arc points between two DXF vertices
- `save_layers_as_shapefiles(layers, output_dir)` — write extract_dxf_layers() output to organized shapefiles

### DXF 3DSOLID Extraction
- `extract_3dsolids(dxf_path, crs, *, layers, bottom_face)` — extract 3DSOLID entities as georeferenced 2D polygon GeoDataFrames (convex hull of ACIS vertices). Auto-applies GEODATA offset if needed.
- `solid3d_to_circle(entity, diameter, *, vertex_index, resolution)` — convert cylindrical 3DSOLID to (center Point, circle Polygon). `vertex_index=-1` (default) or `"midpoint"`. Diameter must be provided (not derivable from ACIS data).

### DXF Document Creation
- `new_dxf_document(version)` — new ezdxf Drawing with proper CAD headers, styles, linetypes
- `ensure_layer(doc, name, color, linetype)` — create layer if missing

### DXF Conversion
- `shapefile_to_dxf(shp_path, dxf_path, *, template_dxf_path, layer, od_table_name, od_schema, od_value_columns, circle_radius_m, mtext_content_fn)` — convert shapefile to DXF with optional Map OD and labels

### DXF Map Object Data (AutoCAD Map OD)
- `encode_od_1004(schema, values)` — encode attribute values to Map OD binary format
- `attach_od_to_entity(doc, entity, table_handle, record_index, binary_1004)` — attach OD to any DXF entity
- `get_table_handle_by_name(doc, table_name)` — find OD table handle by name

### Geometry Utilities
- `repair_geometry(geom, *, context)` — validate and repair a single geometry. Handles self-intersections, GeometryCollection results from make_valid(), extracts Polygon(s) from compound results. Prints warnings when repair is needed. **Used automatically in all DXF extraction — but call explicitly when building geometries from coordinates or dissolving.**
- `remove_inner_rings(geom)` — remove holes from Polygon/MultiPolygon
- `make_valid_gdf(gdf)` — repair all invalid geometries in a GeoDataFrame (returns copy)
- `subtract_geometries(base_gdf, subtract_gdf)` — set difference: base minus union of subtract
- `subtract_smaller_overlaps(gdf)` — where polygons overlap, subtract smaller from larger
- `load_and_union(path, crs)` — load shapefile, union all geometries; returns (union_geom, gdf)
- `find_column(gdf, candidates)` — find first matching column name from candidate list
- `morphological_filter(gdf, min_area_ha, buffer_distance, remove_holes)` — buffer-dissolve-buffer polygon cleanup
- `distance_to_nearest(gdf, reference_gdf, column_name)` — add min distance to nearest reference feature
- `points_with_buffers(data, crs, *, buffer_col, buffer_factor)` — create points from coordinate dicts + optional buffer union

### Line-to-Polygon Operations
- `extend_line(line, distance, *, start=True, end=True)` — extend LineString from one or both endpoints in line direction
- `snap_endpoints(lines, tolerance)` — snap nearby LineString endpoints using cKDTree clustering
- `lines_to_polygon(lines, *, extend=0, snap_tolerance=0, mode="outer")` — pipeline: snap + extend + node + polygonize + union. `mode="outer"` returns exterior only, `mode="all"` returns union of all cells

### Reporting
- `markdown_table(headers, rows, align, number_format, max_col_width)` — fixed-width markdown table that aligns in raw view
- `area_report(layers, *, intersect_with, category_gdf, category_col, crs)` — full markdown area report
- `area_by_category(target_gdf, category_gdf, category_col)` — intersection areas grouped by category
- `intersection_areas(geom, parcels_gdf, *, label_col)` — intersection area per parcel

### OSM Data (import from submodule)
- `from gis_utils.osm import download_osm_polygons` — download polygon features from OpenStreetMap Overpass API
- `from gis_utils.osm import bbox_from_shapefile` — get WGS84 bounding box from a shapefile

### ALKIS Flurstück Lookup
- `find_flurstuecke(state, *, gemarkung, flur, nummern, ...)` — find Flurstücke by Gemarkung/Flur/Nummer. State: 'sh' or 'mv'. Parses "78/2" automatically. Uses WFS stored queries + client-side filtering.
- Requires either `gemarkung_schluessel` (for server-side stored query) or `extent`/`input_boundary` (for spatial download + client-side filter)
- Profiles in `alkis_profiles.yaml` map state codes to recipes

### Recipes (Source Profiles)
- `load_recipe(name, project_dir=None)` — load a source recipe by name (searches project `sources/` then library defaults)
- `list_recipes(project_dir=None, search=None)` — find available recipes by name/description/tags (also searches within multi-layer titles/tags)
- `apply_attribute_mappings(gdf, mappings)` — add `_lbl` label columns from value maps
- `run_multi_layer_recipe(recipe, layer_aliases, ...)` — download selected layers from a multi-layer recipe
- `check_recipe_layers(recipe)` — compare multi-layer recipe against live WFS GetCapabilities
- Shipped single-layer recipes: `mv_bodenschaetzung` (Bodenschätzwerte WMS), `osm_siedlungsflaechen`, `osm_wohngebaeude`
- Shipped multi-layer recipes:
  - `sh_uwat` (SH Umwelt-Atlas: 27 layers — Boden, Erosion, Geologie, Wasser, Landwirtschaft)
  - `sh_alkis` (SH ALKIS vereinfacht: Flurstücke, Katasterbezirke, Verwaltungseinheiten, Gebäude, Nutzung — with query_fields + stored_queries for Flurstück lookup)
  - `mv_bodengeologie` (MV Bodengeologie: 7 layers — Moore, Feldkapazität, Nitrat, etc.)
  - `mv_alkis` (MV ALKIS detailed: 20 layers — Gebäude, Flurstücke, Nutzungen, Verwaltung, Schutzgebiete)
  - `mv_alkis_vereinf` (MV ALKIS vereinfacht: same AdV standard as SH — for Flurstück lookup)
- Multi-layer recipes define `layers:` dict with per-layer config (wfs_layer, title, tags, column_mapping, attribute_mappings). In workflow.yaml: `layers: [alias1, alias2]` + `output_dir:` instead of `output:`
- Recipes can define `query_fields` (friendly→WFS field name mapping) and `stored_queries` for server-side filtering
- WFS download supports `filter` for client-side attribute filtering and `stored_query`/`stored_query_params` for server-side queries
- CLI: `gis-workflow check-recipes [recipe_name]` — compares hardcoded layers vs live endpoint
- Recipe YAMLs define: connection (URL, layer, CRS), detection mode, attribute value mappings, column renaming, post-processing steps, optional Python hooks
- When a user wants a WMS/WFS source that has no recipe yet:
  1. Query GetCapabilities to discover available layers
  2. Sample GetFeatureInfo at multiple points to discover attribute fields and values
  3. Discuss naming and description with the user
  4. Create a recipe YAML in the project's `sources/` directory (or add to library)
  5. Research and populate attribute value mappings from official documentation

### Heavy/Specialized (import from submodule)
- `run_recipe(recipe, input_boundary=..., output_path=...)` — highest-level API: auto-dispatches to WFS download or WMS vectorization based on recipe connection type (single-layer)
- `from gis_utils.wfs import download` — WFS direct vector download (no raster conversion needed)
- `from gis_utils.wms import run` — WMS download, line/area detection, vectorization to GeoPackage/shapefile. Accepts `recipe="name"` for predefined sources.
- `from gis_utils.grass import main` — GRASS GIS raster skeletonization to centerline GeoJSON

### Workflow Runner
- CLI: `gis-workflow run [project_dir]`, `gis-workflow run --dry-run`, `gis-workflow run --step "Name"`
- Init: `gis-workflow init [project_dir]` — creates workflow.yaml, scripts/, CLAUDE.md

### Templates
Built-in workflow templates — reusable processing patterns invoked via `template:` in workflow.yaml. Templates run in-process (no subprocess), are faster than scripts, and encode proven conversion workflows.

Available templates:
- `dxf_extract` — extract DXF layers to GeoPackage. Params: `dxf`, `layers`, `crs`, `strip_zone`
- `dxf_lines_to_polygon` — convert DXF lines to closed polygon via extend + polygonize. Params: `dxf`, `layer`, `crs`, `strip_zone`, `extend`, `snap_tolerance`, `mode`
- `verification_dxf` — write original DXF lines + derived polygon to DXF for visual QA. Params: `dxf`, `layer`, `crs`, `strip_zone`, `polygon`

Usage in workflow.yaml:
```yaml
- name: My Step
  template: dxf_lines_to_polygon
  params:
    dxf: path/to/file.dxf
    layer: LAYER_NAME
    crs: "EPSG:25833"
    strip_zone: true
    extend: 10.0
  output: output/result.gpkg
```

Discovery: `gis-workflow catalog --search template` or `from gis_utils.templates import list_templates`

### Cookbook: DXF lines → polygon
**When:** DXF has polylines that form a boundary but don't connect (gaps at endpoints)
**Symptoms:** Layer has LineStrings, not closed Polygons. Endpoints are close but don't touch.
**Steps:**
1. `extract_dxf_layers()` → get LineStrings
2. `strip_utm_zone_prefix()` if coordinates have 32/33 prefix
3. `lines_to_polygon(lines, extend=10, snap_tolerance=1.0)`
4. If result area is too small → user closes large gaps manually in AutoCAD → re-run
5. Verification: write original lines + result to DXF for visual comparison

Or use the `dxf_lines_to_polygon` template directly in workflow.yaml — no script needed.

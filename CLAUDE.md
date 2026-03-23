# gis_utils

GIS/CAD utility library. Installed as editable in conda env `gis`:
```
pip install -e ~/dev/Gunther-Schulz/gis_utils
```

## API Quick Reference

All common functions importable from top level: `from gis_utils import ...`

### DXF Extraction
- `extract_dxf_layers(dxf_path, crs, *, layers, exclude_layers, circles_as_points, process_blocks)` — extract all geometry from DXF, returns `{layer: {geom_type: GeoDataFrame}}`
- `extract_dxf_circles(dxf_path, crs, *, layers)` — circle centers as Point GeoDataFrame with `radius` column
- `lwpolyline_to_coords(entity, arc_points)` — LWPOLYLINE/POLYLINE to coordinate list with bulge interpolation
- `interpolate_bulge_arc(start, end, bulge, num_points)` — arc points between two DXF vertices
- `save_layers_as_shapefiles(layers, output_dir)` — write extract_dxf_layers() output to organized shapefiles

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
- `remove_inner_rings(geom)` — remove holes from Polygon/MultiPolygon
- `make_valid_gdf(gdf)` — repair all invalid geometries in a GeoDataFrame (returns copy)
- `subtract_geometries(base_gdf, subtract_gdf)` — set difference: base minus union of subtract
- `subtract_smaller_overlaps(gdf)` — where polygons overlap, subtract smaller from larger
- `load_and_union(path, crs)` — load shapefile, union all geometries; returns (union_geom, gdf)
- `find_column(gdf, candidates)` — find first matching column name from candidate list
- `morphological_filter(gdf, min_area_ha, buffer_distance, remove_holes)` — buffer-dissolve-buffer polygon cleanup
- `distance_to_nearest(gdf, reference_gdf, column_name)` — add min distance to nearest reference feature
- `points_with_buffers(data, crs, *, buffer_col, buffer_factor)` — create points from coordinate dicts + optional buffer union

### Reporting
- `markdown_table(headers, rows, align, number_format, max_col_width)` — fixed-width markdown table that aligns in raw view
- `area_report(layers, *, intersect_with, category_gdf, category_col, crs)` — full markdown area report
- `area_by_category(target_gdf, category_gdf, category_col)` — intersection areas grouped by category
- `intersection_areas(geom, parcels_gdf, *, label_col)` — intersection area per parcel

### OSM Data (import from submodule)
- `from gis_utils.osm import download_osm_polygons` — download polygon features from OpenStreetMap Overpass API
- `from gis_utils.osm import bbox_from_shapefile` — get WGS84 bounding box from a shapefile

### Heavy/Specialized (import from submodule)
- `from gis_utils.wms import run` — WMS download, line/area detection, vectorization to shapefile
- `from gis_utils.grass import main` — GRASS GIS raster skeletonization to centerline GeoJSON

### Workflow Runner
- CLI: `gis-workflow run [project_dir]`, `gis-workflow --dry-run`, `gis-workflow --step "Name"`
- Init: `gis-workflow init [project_dir]` — creates workflow.yaml, scripts/, CLAUDE.md

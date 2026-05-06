---
name: gis-safety
description: This skill should be used when the user asks to "write code", "implement", "create a script", "add a function", or any code-writing task in a GIS/CAD project that uses gis_utils. Enforces CRS safety, dangerous defaults prevention, and output conventions.
license: MIT
---

## GIS Safety Rules

Apply these rules to ALL code written in gis_utils projects.

### CRITICAL: No dangerous defaults or silent fallbacks

- **CRS**: NEVER default to a specific EPSG code. Different projects use different zones (25832 vs 25833). A wrong CRS silently shifts geometries by hundreds of meters. Always require CRS explicitly.
- **URLs, layer names, file paths**: NEVER hardcode project-specific values as defaults. Require them as parameters or get them from recipes.
- **Any parameter where a wrong default produces valid-looking but incorrect output**: make it required, not optional with a default.
- **Safe defaults are OK**: `timeout=120`, `dissolve=True`, `simplify_tolerance=1.0` — wrong values cause obvious failures, not silent corruption.
- **When in doubt**: require the parameter with no default. An explicit error is always better than silently wrong data.

### Output convention: always single Polygons

When producing GeoDataFrame outputs, always explode MultiPolygons into individual Polygon features (`.explode(index_parts=False)`). MultiPolygons make styling, labeling, and area calculations unreliable in QGIS.

### Alpha stage — no backward compatibility

This library is in alpha. Do not add backward-compatibility shims, deprecated aliases, re-exports of renamed symbols, or any code whose sole purpose is keeping old callers working. When something changes, just change it.

### Project context

When starting work in a GIS project:
1. Read `workflow.yaml` to understand the pipeline, CRS, and data sources
2. If CRS is not apparent from the workflow, ask the user before writing any code
3. Check existing scripts in `scripts/` for conventions used in this project

### Discovery before coding

Before writing code, use the `gis-utils` MCP tools to discover available functions:
- `mcp__gis-utils__catalog` — search the full API
- `mcp__gis-utils__list_recipes` — find data source recipes
- `mcp__gis-utils__list_templates` — find workflow templates
- `mcp__gis-utils__get_function_help` — get detailed function docs

### WMS layers in QGIS — set extent close after add

When adding a WMS raster (DOP, ALKIS, etc.) to a fresh QGIS project: **set the canvas extent to a close-up area (a few km) AFTER `addMapLayer` and after any `refreshAllLayers()`**. Auto-zoom to the layer's full extent (e.g. a Bundesland-wide DOP) usually exceeds the service's scale-denominator limit and the canvas stays blank.

**How to apply:** add the layer → compute target extent (project AOI / city centre + ~5 km box) → `iface.mapCanvas().setExtent(rect)` as the **last** step (or use the `mcp__qgis__set_canvas_extent` MCP tool). If a refresh is needed in between, set the extent twice — once before and once after refresh. Always verify with a screenshot before declaring done.

For German DOP / ALKIS WMS specifically: parse `GetCapabilities` as XML (not regex) to find `<Layer><Name>` plus required `<Style><Name>` — some MV services (e.g. `adv_dop`) require an explicit non-empty `styles=` parameter (use `palette_rgb` for `mv_dop`).

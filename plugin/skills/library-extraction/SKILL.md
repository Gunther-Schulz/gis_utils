---
name: library-extraction
description: This skill should be used when the user asks to "write a script", "add to the workflow", "create a processing step", or when writing project-local Python code in a GIS project that uses gis_utils. Evaluates whether logic should be extracted into the gis_utils library.
license: MIT
---

## Library Extraction

When writing project-local scripts (`scripts/*.py` or inline workflow steps), evaluate whether the logic is reusable.

### Before writing project-local code

1. **Check if it already exists**: use `mcp__gis-utils__catalog` to search for existing functions
2. **If it doesn't exist, ask**: is this operation specific to this project, or would other projects benefit?

### Signs that code should go into gis_utils

- The operation works on standard types (GeoDataFrame, Shapely geometry, DXF entities) — not project-specific data
- You can describe it without mentioning the project name or specific file paths
- Similar logic exists in other project scripts (pattern repetition)
- It's a transformation, filter, conversion, or analysis that's parameterizable

### When extracting to the library

1. **Generalize parameters**: replace hardcoded values (CRS, layer names, file paths, thresholds) with function parameters. No project-specific defaults.
2. **Decompose**: break into small, composable functions. One function per operation. A 50-line script often becomes 2-3 library functions that each do one thing.
3. **Follow existing patterns**: use `mcp__gis-utils__catalog` to find similar functions and match their conventions (parameter naming, return types, docstring format).
4. **Add to the right module**: check existing module structure. Geometry ops → `geometry.py`, DXF ops → `dxf/`, reporting → `reporting.py`, etc.
5. **Update the local script**: replace the inlined logic with a call to the new library function. The script becomes a thin wrapper that passes project-specific parameters.
6. **Update workflow.yaml**: if the extraction enables a template pattern, consider creating a template.
7. **Update catalog.py**: if you added a new submodule, add it to `_MODULES`. New CLI subcommands → add to `_CLI_COMMANDS`. Functions are auto-discovered.

### When to keep it local

- Logic depends on project-specific file layouts or naming conventions
- It's a one-off data cleaning step unlikely to recur
- The operation is too narrow to generalize without over-engineering

### Example

**Local script** has: read shapefile, buffer by 50m, dissolve, clip to boundary, save.

**Extraction**: `buffer` and `dissolve` are already in geopandas. But "buffer by attribute column × factor" is a reusable pattern → extract as `points_with_buffers()`. The clip-to-boundary step is project-specific → keep local.

**Result**: script shrinks from 30 lines to 5 (import + call library function + project-specific clip + save).

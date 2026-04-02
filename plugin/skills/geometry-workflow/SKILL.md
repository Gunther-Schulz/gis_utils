---
name: geometry-workflow
description: This skill should be used when the user asks to "convert geometry", "close gaps", "lines to polygon", "extract boundaries", "find overlaps", "buffer", "dissolve", or any geometry conversion/analysis task in a GIS project. Enforces data analysis before coding.
license: MIT
---

## Geometry Workflow: Analyze Before Coding

For geometry conversion/analysis tasks (lines to polygons, closing gaps, finding boundaries, extracting features, etc.), **always analyze the data before writing any solution code**:

1. **Inspect the data**: count features, types, lengths, orientations
2. **Measure relationships**: gaps between endpoints, connectivity, parallel vs crossing
3. **Ask the user** if the physical interpretation is unclear ("these look like two parallel boundary lines with cross-lines between them — is that right?")
4. **Think: how would a user solve this manually?** What steps would they take in AutoCAD, QGIS, or another GIS application? The manual workflow almost always reveals the simplest automated approach:
   - AutoCAD: extend, trim, close, join, hatch boundary, polyline edit
   - QGIS: buffer, dissolve, merge, polygonize, select by location, field calculator
   - The manual steps translate directly to shapely/geopandas operations
5. **Start with that naive approach FIRST.** Only escalate to computational geometry algorithms (concave hull, alpha shapes, graph traversal) if the simple approach actually fails on the data

### Example

13 disconnected polylines with 5–18 m gaps → a user in AutoCAD would just extend the lines and trim. Don't reach for concave hull or planar graph traversal. Just `extend_line` + `polygonize`. Done.

### Use the MCP tools to find existing functions

Before writing geometry code, check what already exists:
```
mcp__gis-utils__catalog(search="geometry")
mcp__gis-utils__catalog(search="polygon")
mcp__gis-utils__list_templates()
```

The library likely has a function or template for what you need. Don't reimplement.

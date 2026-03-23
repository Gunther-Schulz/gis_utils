#!/usr/bin/env python3
"""
Run inside GRASS session: grass /path/to/loc/PERMANENT --exec python grass_centerlines.py <input_green.tif> <output.geojson>
Reads a binary raster (1=line, 0=background), thins to centerlines (r.thin), vectorizes (r.to.vect),
exports to GeoJSON. Expects GRASS location already created from the same raster (same extent/CRS).
"""
from __future__ import annotations

import sys

def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: grass /path/to/loc/PERMANENT --exec python grass_centerlines.py <input_green.tif> <output.geojson>", file=sys.stderr)
        return 1
    input_tif = sys.argv[1]
    output_geojson = sys.argv[2]

    import grass.script as gs

    gs.message("Importing green mask raster...")
    gs.run_command("r.in.gdal", input=input_tif, output="green_mask", flags="o", overwrite=True)
    gs.run_command("r.null", map="green_mask", setnull="0")  # 0 -> null so r.thin sees only 1s
    gs.message("Thinning (skeletonizing) to centerlines...")
    gs.run_command("r.thin", input="green_mask", output="green_thin", overwrite=True)
    gs.message("Vectorizing centerlines...")
    gs.run_command("r.to.vect", input="green_thin", output="centerlines", type="line", overwrite=True)
    gs.message("Exporting to GeoJSON...")
    gs.run_command("v.out.ogr", input="centerlines", output=output_geojson, format="GeoJSON", overwrite=True)
    gs.message("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

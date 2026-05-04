"""buffer_zones — Concentric ring buffer zones around a source geometry.

Produces ring polygons at named distance bands around a source (point, line,
or polygon) layer.  Optionally intersects the rings with a target layer and
writes a per-zone area report.

Typical use case: planning analyses where a project area's relationship to
infrastructure (Autobahn, Bahnstrecke, Hochspannungsleitung, Gewässer) is
defined by distance bands — e.g. § 35 Abs. 1 Nr. 8 BauGB privileging zone of
200 m around BAB, with sub-band 0–110 m.

Example workflow.yaml::

    - name: BAB Pufferzonen Wölzow
      template: buffer_zones
      params:
        source: Shape/A24_Verkehrsflaeche.shp
        crs: "EPSG:25833"
        zones:
          - name: "0-110m"
            outer_m: 110
          - name: "110-200m"
            inner_m: 110
            outer_m: 200
        target: Shape/Projektfläche.shp        # optional
        report_csv: area_by_bab_zone.csv       # optional
      output: Shape/bab_pufferzonen.gpkg

The output GPKG contains:

- Layer ``zones`` — the ring polygons (one row per zone).
- Layer ``zones_target_intersection`` — only present when ``target`` is given.
  One row per zone × target intersection.

Each layer carries ``name``, ``inner_m``, ``outer_m``, ``area_m2`` columns.
"""

from __future__ import annotations

from pathlib import Path

from gis_utils.templates import register


@register(
    "buffer_zones",
    description=(
        "Concentric ring buffer zones around a source layer (point/line/polygon); "
        "optional target intersection + area report"
    ),
    params=["source", "crs", "zones", "target", "report_csv"],
)
def buffer_zones_template(
    params: dict, project_dir: Path, output_path: Path
) -> bool:
    """Generate concentric buffer ring zones around a source layer.

    Params:
        source: Path to source layer (Shapefile / GeoPackage / GeoJSON).
            Geometries are unioned into one geometry before buffering.
        crs: Projected CRS in metres (e.g. ``"EPSG:25833"``).
        zones: List of zone definitions, each a dict with:
            - ``name`` (str): zone label.
            - ``outer_m`` (float): outer distance in CRS units.
            - ``inner_m`` (float, optional, default 0): inner distance.
        target (optional): Path to target layer.  If given, the rings are
            intersected with the target and a second GPKG layer
            ``zones_target_intersection`` is written.
        report_csv (optional): Path to CSV file for the area report.
            Written when the rings produce non-empty geometries.
            Columns: ``Zone | Ring (m²) | Ring (ha)``, plus
            ``Target ∩ (m²) | Target ∩ (ha)`` when ``target`` is given.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    from gis_utils import buffer_ring_zones, markdown_table

    source_path = project_dir / params["source"]
    crs = params["crs"]
    zones_cfg = params["zones"]
    target_param = params.get("target")
    report_csv = params.get("report_csv")

    if not source_path.exists():
        print(f"  [ERROR] source not found: {source_path}")
        return False

    src_gdf = gpd.read_file(source_path)
    if src_gdf.empty:
        print(f"  [ERROR] source has no features: {source_path}")
        return False
    if src_gdf.crs is None:
        src_gdf = src_gdf.set_crs(crs)
    elif str(src_gdf.crs) != crs:
        src_gdf = src_gdf.to_crs(crs)

    source_geom = unary_union(src_gdf.geometry.values)
    if source_geom is None or source_geom.is_empty:
        print(f"  [ERROR] source geometry is empty after union")
        return False

    rings = buffer_ring_zones(source_geom, zones_cfg)
    if not rings:
        print(f"  [ERROR] no non-empty ring zones produced")
        return False

    zone_records = [
        {
            "name": meta["name"],
            "inner_m": meta["inner_m"],
            "outer_m": meta["outer_m"],
            "area_m2": meta["area_m2"],
            "geometry": geom,
        }
        for meta, geom in rings
    ]
    zones_gdf = gpd.GeoDataFrame(zone_records, crs=crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write GPKG (multi-layer support); single-layer fallback for .shp.
    out_ext = output_path.suffix.lower()
    if out_ext in (".gpkg", ".geopackage"):
        zones_gdf.to_file(output_path, driver="GPKG", layer="zones")
    else:
        zones_gdf.to_file(output_path)
    print(f"  Zones: {len(zones_gdf)} ring polygons → {output_path}")

    # --- Intersection with target ---
    intersection_gdf = None
    if target_param:
        target_path = project_dir / target_param
        if not target_path.exists():
            print(f"  [WARN] target not found, skipping intersection: {target_path}")
        else:
            tgt_gdf = gpd.read_file(target_path)
            if tgt_gdf.crs is None:
                tgt_gdf = tgt_gdf.set_crs(crs)
            elif str(tgt_gdf.crs) != crs:
                tgt_gdf = tgt_gdf.to_crs(crs)
            tgt_union = unary_union(tgt_gdf.geometry.values)

            int_records = []
            for meta, ring in rings:
                inter = ring.intersection(tgt_union)
                if inter.is_empty:
                    continue
                int_records.append({
                    "name": meta["name"],
                    "inner_m": meta["inner_m"],
                    "outer_m": meta["outer_m"],
                    "area_m2": float(inter.area),
                    "geometry": inter,
                })
            intersection_gdf = gpd.GeoDataFrame(int_records, crs=crs)

            if not intersection_gdf.empty and out_ext in (".gpkg", ".geopackage"):
                intersection_gdf.to_file(
                    output_path, driver="GPKG",
                    layer="zones_target_intersection",
                )
                print(
                    f"  Intersection: {len(intersection_gdf)} polygons "
                    f"(layer 'zones_target_intersection')"
                )

    # --- Report ---
    csv_rows = []
    md_rows = []
    if intersection_gdf is None or intersection_gdf.empty:
        for meta, _ in rings:
            m2 = round(meta["area_m2"])
            ha = round(meta["area_m2"] / 10_000, 4)
            csv_rows.append([meta["name"], m2, ha])
            md_rows.append([meta["name"], f"{m2:,}", f"{ha:.4f}"])
        headers = ["Zone", "Ring (m²)", "Ring (ha)"]
    else:
        int_by_zone = {
            r["name"]: r["area_m2"]
            for _, r in intersection_gdf.iterrows()
        }
        for meta, _ in rings:
            tarea = int_by_zone.get(meta["name"], 0.0)
            ring_m2 = round(meta["area_m2"])
            ring_ha = round(meta["area_m2"] / 10_000, 4)
            tgt_m2 = round(tarea)
            tgt_ha = round(tarea / 10_000, 4)
            csv_rows.append([meta["name"], ring_m2, ring_ha, tgt_m2, tgt_ha])
            md_rows.append([
                meta["name"],
                f"{ring_m2:,}", f"{ring_ha:.4f}",
                f"{tgt_m2:,}", f"{tgt_ha:.4f}",
            ])
        headers = [
            "Zone", "Ring (m²)", "Ring (ha)",
            "Target ∩ (m²)", "Target ∩ (ha)",
        ]

    print()
    print(markdown_table(headers, md_rows))

    if report_csv:
        csv_path = project_dir / report_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows, columns=headers).to_csv(csv_path, index=False)
        print(f"  Report: {csv_path}")

    return True

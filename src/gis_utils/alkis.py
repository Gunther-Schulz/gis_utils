"""Convenience API for ALKIS Flurstück lookups across German states.

Provides a human-friendly interface on top of the generic WFS/recipe system.
State-specific endpoints and field names are defined in recipe YAMLs;
the mapping of state codes to recipes lives in alkis_profiles.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_profiles() -> dict[str, dict[str, str]]:
    """Load state → recipe/layer mapping from alkis_profiles.yaml."""
    profiles_path = Path(__file__).parent / "alkis_profiles.yaml"
    with open(profiles_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_nummer(nummer: str) -> tuple[str, str]:
    """Parse a Flurstück number like '78/2' into (zaehler, nenner).

    Supports formats: '78/2', '78', '123/45'.
    """
    if "/" in nummer:
        parts = nummer.split("/", 1)
        return parts[0].strip(), parts[1].strip()
    return nummer.strip(), ""


def find_flurstuecke(
    state: str,
    *,
    gemarkung: str | None = None,
    flur: str | None = None,
    nummern: list[str] | None = None,
    oids: list[str] | None = None,
    gemeinde: str | None = None,
    gemarkung_schluessel: str | None = None,
    gemeinde_schluessel: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    input_boundary: "Path | str | None" = None,
    buffer_m: float | None = None,
    crs: str | None = None,
    output_path: "Path | str | None" = None,
    project_dir: "Path | str | None" = None,
) -> "gpd.GeoDataFrame":
    """Find Flurstücke by Gemarkung, Flur, Nummer(n), or OID.

    Two modes of operation:
    1. With gemarkung_schluessel or gemeinde_schluessel: uses WFS stored
       queries for efficient server-side filtering.
    2. With extent/input_boundary: downloads all Flurstücke in the area,
       then filters client-side by gemarkung/flur/nummer/oid.

    OID filter is client-side because MV ave:Flurstueck rejects
    server-side OGC filters on `oid` (returns HTTP 400). The bbox path
    is the workable path; provide extent or input_boundary alongside oids.

    Args:
        state: Bundesland code ('sh', 'mv', etc.).
        gemarkung: Gemarkung name for client-side filtering.
        flur: Flur number for client-side filtering.
        nummern: Flurstück numbers (e.g. ['78/2']). Parsed automatically.
        oids: ALKIS OIDs (e.g. ['DEMVAL04000wCbubFL']). Client-side filtered
            after a bbox download — requires extent or input_boundary.
        gemeinde: Gemeinde name for client-side filtering.
        gemarkung_schluessel: Gemarkungsnummer for stored query (e.g. '010266').
        gemeinde_schluessel: Gemeindeschlüssel for stored query (e.g. '01061108').
        extent: (minx, miny, maxx, maxy) bounding box in crs.
        input_boundary: Shapefile to derive extent from.
        buffer_m: Buffer in meters around input_boundary/extent.
        crs: CRS for output (required).
        output_path: Optional output file path (.gpkg recommended).
        project_dir: Project directory for recipe search.

    Returns:
        GeoDataFrame with matching Flurstücke.
    """
    import geopandas as gpd
    from gis_utils.recipes import load_recipe
    from gis_utils.wfs import download

    profiles = _load_profiles()
    state_lower = state.lower()
    if state_lower not in profiles:
        raise ValueError(
            f"Unknown state '{state}'. Available: {list(profiles.keys())}"
        )
    profile = profiles[state_lower]
    recipe_name = profile["recipe"]
    layer_alias = profile["layer"]

    recipe = load_recipe(recipe_name, project_dir=Path(project_dir) if project_dir else None)
    layer_recipe = recipe.get_layer_recipe(layer_alias)
    layer_cfg = recipe.layers[layer_alias]

    if not crs:
        raise ValueError("crs is required (e.g. 'EPSG:25832').")

    query_fields = layer_cfg.get("query_fields", {})
    stored_queries = layer_cfg.get("stored_queries", {})
    conn = layer_recipe.connection

    # --- Resolve extent from input_boundary ---
    _extent = extent
    if _extent is None and input_boundary is not None:
        boundary_gdf = gpd.read_file(input_boundary).to_crs(crs)
        b = boundary_gdf.total_bounds
        _extent = tuple(b)
    if _extent is not None and buffer_m:
        _extent = (
            _extent[0] - buffer_m, _extent[1] - buffer_m,
            _extent[2] + buffer_m, _extent[3] + buffer_m,
        )

    # --- Determine download strategy ---
    stored_query_id = None
    stored_query_params = None

    if gemarkung_schluessel and "by_gemarkung" in stored_queries:
        sq = stored_queries["by_gemarkung"]
        stored_query_id = sq["id"]
        stored_query_params = {sq["param_field"]: gemarkung_schluessel}
    elif gemeinde_schluessel and "by_gemeinde" in stored_queries:
        sq = stored_queries["by_gemeinde"]
        stored_query_id = sq["id"]
        stored_query_params = {sq["param_field"]: gemeinde_schluessel}
    elif _extent is None:
        raise ValueError(
            "Either provide gemarkung_schluessel/gemeinde_schluessel for "
            "server-side query, or extent/input_boundary for spatial query."
        )

    # --- Build client-side filter ---
    client_filter = {}
    if gemarkung:
        client_filter[query_fields.get("gemarkung", "gemarkung")] = gemarkung
    if flur:
        client_filter[query_fields.get("flur", "flur")] = flur
    if gemeinde:
        client_filter[query_fields.get("gemeinde", "gemeinde")] = gemeinde

    # --- Download ---
    gdf = download(
        url=conn.get("url", ""),
        layer=conn.get("layer", ""),
        crs=crs,
        version=conn.get("version", "2.0.0"),
        extent=_extent if not stored_query_id else None,
        output_path=None,
        recipe=layer_recipe,
        recipe_dir=project_dir,
        stored_query=stored_query_id,
        stored_query_params=stored_query_params,
        filter=client_filter if client_filter else None,
    )

    # --- Client-side filter for specific OIDs ---
    if oids and len(gdf) > 0:
        oid_col = query_fields.get("oid", "oid")
        if oid_col not in gdf.columns:
            raise ValueError(
                f"oids filter requested but column '{oid_col}' not in WFS response. "
                f"Available columns: {list(gdf.columns)}"
            )
        gdf = gdf[gdf[oid_col].isin(oids)].copy()

    # --- Client-side filter for specific Flurstück numbers ---
    if nummern and len(gdf) > 0:
        z_col = query_fields.get("zaehler", "flstnrzae")
        n_col = query_fields.get("nenner", "flstnrnen")
        parsed = [_parse_nummer(n) for n in nummern]
        mask = None
        for zaehler, nenner in parsed:
            # Normalize: strip trailing .0 from float-stored values
            z_norm = gdf[z_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            m = z_norm == zaehler
            if nenner:
                n_norm = gdf[n_col].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                m = m & (n_norm == nenner)
            mask = m if mask is None else (mask | m)
        if mask is not None:
            gdf = gdf[mask].copy()

    if len(gdf) == 0:
        parts = []
        if gemarkung:
            parts.append(f"Gemarkung={gemarkung}")
        if flur:
            parts.append(f"Flur={flur}")
        if nummern:
            parts.append(f"Nummern={nummern}")
        if oids:
            parts.append(f"OIDs={oids}")
        if gemeinde:
            parts.append(f"Gemeinde={gemeinde}")
        print(f"[alkis] Warning: no Flurstücke found for {', '.join(parts)}")
    else:
        print(f"[alkis] Found {len(gdf)} Flurstück(e)")

    # Write final output
    if output_path is not None and len(gdf) > 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        print(f"[alkis] Written: {output_path}")

    return gdf

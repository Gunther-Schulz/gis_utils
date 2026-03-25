"""Recipe system for WMS/WFS source processing.

A recipe bundles everything needed to vectorize a specific data source:
connection info, detection settings, attribute mappings, column renaming,
and optional pre/post processing steps.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Recipe:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    connection: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)
    attribute_mappings: dict[str, dict[str, str]] = field(default_factory=dict)
    column_mapping: dict[str, str] = field(default_factory=dict)
    post_processing: list[dict[str, Any]] = field(default_factory=list)
    hooks: str | None = None
    exclude_tags: dict[str, str] = field(default_factory=dict)
    layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    _source_path: Path | None = field(default=None, repr=False)

    @property
    def is_multi_layer(self) -> bool:
        return bool(self.layers)

    def layer_aliases(self) -> list[str]:
        """Return all layer aliases defined in this recipe."""
        return list(self.layers.keys())

    def get_layer_recipe(self, alias: str) -> "Recipe":
        """Return a single-layer Recipe for the given alias.

        Merges per-layer config (attribute_mappings, column_mapping, etc.)
        with the parent recipe's connection info, producing a Recipe that
        works with the existing run_recipe / wfs.download pipeline.
        """
        if alias not in self.layers:
            raise KeyError(
                f"Layer '{alias}' not found in recipe '{self.name}'. "
                f"Available: {self.layer_aliases()}"
            )
        layer_cfg = self.layers[alias]
        conn = dict(self.connection)
        conn["layer"] = layer_cfg.get("wfs_layer", alias)
        return Recipe(
            name=f"{self.name}:{alias}",
            description=layer_cfg.get("title", alias),
            tags=layer_cfg.get("tags", []),
            connection=conn,
            detection=layer_cfg.get("detection", self.detection),
            attribute_mappings=layer_cfg.get("attribute_mappings", {}),
            column_mapping=layer_cfg.get("column_mapping", {}),
            post_processing=layer_cfg.get("post_processing", self.post_processing),
            hooks=layer_cfg.get("hooks", self.hooks),
            exclude_tags=layer_cfg.get("exclude_tags", {}),
            _source_path=self._source_path,
        )


def _library_sources_dir() -> Path:
    """Return path to the shipped sources/ directory inside the package."""
    if sys.version_info >= (3, 9):
        ref = importlib.resources.files("gis_utils") / "sources"
        # For editable installs, this is a real Path; for installed, a Traversable
        return Path(str(ref))
    # Fallback for older Python
    return Path(__file__).parent / "sources"


def _sources_dirs(project_dir: Path | None = None) -> list[Path]:
    """Return search directories in priority order: project-local first, then library."""
    dirs = []
    if project_dir is not None:
        local = Path(project_dir) / "sources"
        if local.is_dir():
            dirs.append(local)
    lib = _library_sources_dir()
    if lib.is_dir():
        dirs.append(lib)
    return dirs


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_recipe(data: dict, source_path: Path | None = None) -> Recipe:
    return Recipe(
        name=data.get("name", ""),
        description=data.get("description", ""),
        tags=data.get("tags", []),
        connection=data.get("connection", {}),
        detection=data.get("detection", {}),
        attribute_mappings=data.get("attribute_mappings", {}),
        column_mapping=data.get("column_mapping", {}),
        post_processing=data.get("post_processing", []),
        hooks=data.get("hooks"),
        exclude_tags=data.get("exclude_tags", {}),
        layers=data.get("layers", {}),
        _source_path=source_path,
    )


def load_recipe(name: str, project_dir: Path | None = None) -> Recipe:
    """Load a source recipe by name.

    Searches project_dir/sources/ first, then library-shipped sources/.
    The name can match either the filename (without .yaml) or the 'name' field
    inside the YAML.
    """
    # First try exact filename match
    for d in _sources_dirs(project_dir):
        candidate = d / f"{name}.yaml"
        if candidate.is_file():
            return _parse_recipe(_load_yaml(candidate), candidate)

    # Then search by 'name' field inside YAMLs
    for d in _sources_dirs(project_dir):
        for f in sorted(d.glob("*.yaml")):
            data = _load_yaml(f)
            if data.get("name") == name:
                return _parse_recipe(data, f)

    available = [r.name for r in list_recipes(project_dir)]
    raise FileNotFoundError(
        f"Recipe '{name}' not found. Available: {available}"
    )


def list_recipes(
    project_dir: Path | None = None,
    search: str | None = None,
) -> list[Recipe]:
    """List available recipes, optionally filtered by search term.

    Search matches against name, description, and tags (case-insensitive).
    """
    seen_names: set[str] = set()
    recipes: list[Recipe] = []

    for d in _sources_dirs(project_dir):
        for f in sorted(d.glob("*.yaml")):
            data = _load_yaml(f)
            r = _parse_recipe(data, f)
            # Project-local recipes override library ones with same name
            if r.name in seen_names:
                continue
            seen_names.add(r.name)
            recipes.append(r)

    if search:
        term = search.lower()
        recipes = [r for r in recipes if _recipe_matches(r, term)]

    return recipes


def _recipe_matches(recipe: Recipe, term: str) -> bool:
    """Check if a recipe matches a search term (name, description, tags, layer info)."""
    if (term in recipe.name.lower()
            or term in recipe.description.lower()
            or any(term in t.lower() for t in recipe.tags)):
        return True
    for alias, cfg in recipe.layers.items():
        if (term in alias.lower()
                or term in cfg.get("title", "").lower()
                or any(term in t.lower() for t in cfg.get("tags", []))):
            return True
    return False


def resolve_connection(recipe: Recipe) -> dict[str, str]:
    """Resolve recipe connection to kwargs dict.

    Returns dict with keys depending on connection type:
    - WMS: {type, wms_url, wms_layer, crs}
    - WFS: {type, wfs_url, layer, crs, version}
    """
    conn = recipe.connection
    url = conn.get("url")
    conn_type = conn.get("type", "wms").lower()

    # OSM recipes don't need a URL
    if conn_type == "osm":
        return {"type": "osm"}

    if not url and conn.get("qgis_name"):
        from gis_utils.qgis_catalog import qgis_connection
        qc = qgis_connection(conn["qgis_name"])
        if qc is None:
            raise ValueError(
                f"Recipe '{recipe.name}': QGIS connection "
                f"'{conn['qgis_name']}' not found"
            )
        url = qc.url

    if not url:
        raise ValueError(
            f"Recipe '{recipe.name}': no url or qgis_name in connection"
        )

    result: dict[str, str] = {"type": conn_type}
    if conn_type == "wfs":
        result["wfs_url"] = url
        if conn.get("layer"):
            result["layer"] = conn["layer"]
        if conn.get("crs"):
            result["crs"] = conn["crs"]
        if conn.get("version"):
            result["version"] = conn["version"]
    else:
        result["wms_url"] = url
        if conn.get("layer"):
            result["wms_layer"] = conn["layer"]
        if conn.get("crs"):
            result["crs"] = conn["crs"]
    return result


def apply_attribute_mappings(
    gdf: "gpd.GeoDataFrame",
    mappings: dict[str, dict[str, str]],
) -> "gpd.GeoDataFrame":
    """Add label columns for mapped attribute values.

    For each column in mappings, adds a '{col}_lbl' column with the
    human-readable label. The raw column is preserved.
    """
    for col, value_map in mappings.items():
        if col not in gdf.columns:
            continue
        gdf[f"{col}_lbl"] = gdf[col].astype(str).str.strip().map(
            lambda v, m=value_map: m.get(v, "")
        )
    return gdf


def apply_column_mapping(
    gdf: "gpd.GeoDataFrame",
    mapping: dict[str, str],
    is_shapefile: bool = False,
) -> "gpd.GeoDataFrame":
    """Rename columns according to mapping. Truncates to 10 chars for shapefiles."""
    rename = {}
    for old, new in mapping.items():
        if old in gdf.columns:
            if is_shapefile and len(new) > 10:
                new = new[:10]
            rename[old] = new
    if rename:
        gdf = gdf.rename(columns=rename)
    return gdf


# Allowed post-processing functions (name -> import path)
_POST_PROCESSING_FUNCTIONS = {
    "make_valid_gdf": "gis_utils.geometry",
    "morphological_filter": "gis_utils.geometry",
    "subtract_smaller_overlaps": "gis_utils.geometry",
    "subtract_geometries": "gis_utils.geometry",
    "remove_inner_rings": "gis_utils.geometry",
}


def apply_post_processing(
    gdf: "gpd.GeoDataFrame",
    steps: list[dict[str, Any]],
) -> "gpd.GeoDataFrame":
    """Apply declarative post-processing steps to a GeoDataFrame.

    Each step is a dict with one key (function name) and value (kwargs dict).
    Only functions in the allowed vocabulary are permitted.
    """
    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"Invalid post-processing step: {step}")
        func_name, kwargs = next(iter(step.items()))
        if func_name not in _POST_PROCESSING_FUNCTIONS:
            raise ValueError(
                f"Unknown post-processing function: '{func_name}'. "
                f"Allowed: {list(_POST_PROCESSING_FUNCTIONS.keys())}"
            )
        module_path = _POST_PROCESSING_FUNCTIONS[func_name]
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)

        if not isinstance(kwargs, dict):
            kwargs = {}

        # Special handling: some functions operate on geometries, not GDFs
        if func_name == "remove_inner_rings":
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].apply(func, **kwargs)
        else:
            gdf = func(gdf, **kwargs)

    return gdf


def load_and_run_hook(
    hook_filename: str,
    phase: str,
    gdf: "gpd.GeoDataFrame",
    project_dir: Path | None = None,
) -> "gpd.GeoDataFrame":
    """Load a Python hook file and run its pre_process or post_process function.

    Hook files are searched in sources/hooks/ directories (project-local first).
    phase must be 'pre_process' or 'post_process'.
    """
    if phase not in ("pre_process", "post_process"):
        raise ValueError(f"Invalid hook phase: {phase}")

    hook_path = None
    for d in _sources_dirs(project_dir):
        candidate = d / "hooks" / hook_filename
        if candidate.is_file():
            hook_path = candidate
            break

    if hook_path is None:
        print(f"  Warning: hook file '{hook_filename}' not found, skipping")
        return gdf

    spec = importlib.util.spec_from_file_location(
        f"recipe_hook_{hook_path.stem}", hook_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    func = getattr(mod, phase, None)
    if func is None:
        return gdf

    return func(gdf)


def run_recipe(
    recipe: str | Recipe,
    *,
    input_boundary: "Path | str | None" = None,
    extent: tuple[float, float, float, float] | None = None,
    output_path: "Path | str | None" = None,
    recipe_dir: "Path | str | None" = None,
    **kwargs,
) -> "gpd.GeoDataFrame":
    """Run a recipe, automatically dispatching to WFS, WMS, or OSM based on connection type.

    This is the simplest way to use a recipe — just provide the recipe name
    and where you want the output.

    Args:
        recipe: Recipe name (string) or Recipe object.
        input_boundary: Shapefile/GeoPackage to derive extent from.
        extent: (minx, miny, maxx, maxy) bounding box.
        output_path: Output file (.gpkg recommended, .shp supported).
        recipe_dir: Project directory for recipe search.
        **kwargs: Additional arguments passed to the underlying function.

    Returns:
        GeoDataFrame with the result.
    """
    if isinstance(recipe, str):
        _recipe = load_recipe(recipe, project_dir=Path(recipe_dir) if recipe_dir else None)
    else:
        _recipe = recipe

    conn = resolve_connection(_recipe)
    conn_type = conn.get("type", "wms")

    if conn_type == "osm":
        gdf = _run_osm_recipe(
            _recipe, input_boundary=input_boundary, extent=extent,
            output_path=output_path, recipe_dir=recipe_dir, **kwargs,
        )
    elif conn_type == "wfs":
        from gis_utils.wfs import download
        wfs_kwargs = {
            "url": conn["wfs_url"],
            "layer": conn.get("layer", ""),
            "extent": extent,
            "input_boundary": input_boundary,
            "output_path": output_path,
            "crs": conn.get("crs"),
            "version": conn.get("version", "1.1.0"),
            "recipe": _recipe,
            "recipe_dir": recipe_dir,
        }
        wfs_kwargs.update(kwargs)  # caller overrides recipe defaults
        gdf = download(**wfs_kwargs)
    else:
        from gis_utils.wms import run
        wms_kwargs = {
            "extent": extent,
            "output_path": output_path,
            "input_boundary": input_boundary,
            "recipe": _recipe,
            "recipe_dir": recipe_dir,
        }
        wms_kwargs.update(kwargs)
        gdf = run(**wms_kwargs)
    return gdf


def run_multi_layer_recipe(
    recipe: str | Recipe,
    layer_aliases: list[str],
    *,
    input_boundary: "Path | str | None" = None,
    extent: tuple[float, float, float, float] | None = None,
    output_dir: "Path | str | None" = None,
    recipe_dir: "Path | str | None" = None,
    **kwargs,
) -> dict[str, "gpd.GeoDataFrame"]:
    """Run selected layers from a multi-layer recipe.

    Args:
        recipe: Recipe name or Recipe object (must be multi-layer).
        layer_aliases: List of layer aliases to download.
        input_boundary: Shapefile/GeoPackage to derive extent from.
        extent: (minx, miny, maxx, maxy) bounding box.
        output_dir: Directory for output files (one .gpkg per layer).
        recipe_dir: Project directory for recipe search.
        **kwargs: Additional arguments passed to the underlying function.

    Returns:
        Dict mapping alias to GeoDataFrame.
    """
    if isinstance(recipe, str):
        _recipe = load_recipe(recipe, project_dir=Path(recipe_dir) if recipe_dir else None)
    else:
        _recipe = recipe

    if not _recipe.is_multi_layer:
        raise ValueError(f"Recipe '{_recipe.name}' is not a multi-layer recipe")

    available = _recipe.layer_aliases()
    unknown = [a for a in layer_aliases if a not in available]
    if unknown:
        raise KeyError(
            f"Unknown layers {unknown} in recipe '{_recipe.name}'. "
            f"Available: {available}"
        )

    _output_dir = Path(output_dir) if output_dir else None
    if _output_dir:
        _output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for alias in layer_aliases:
        layer_recipe = _recipe.get_layer_recipe(alias)
        out_path = _output_dir / f"{alias}.gpkg" if _output_dir else None
        results[alias] = run_recipe(
            layer_recipe,
            input_boundary=input_boundary,
            extent=extent,
            output_path=out_path,
            recipe_dir=recipe_dir,
            **kwargs,
        )
    return results


def check_recipe_layers(
    recipe: str | Recipe,
    *,
    project_dir: "Path | str | None" = None,
) -> dict[str, list[str]]:
    """Compare a multi-layer recipe against live WFS GetCapabilities.

    Returns dict with keys:
        missing: layers in recipe but not on endpoint
        new: layers on endpoint but not in recipe
        ok: layers in both
    """
    if isinstance(recipe, str):
        _recipe = load_recipe(recipe, project_dir=Path(project_dir) if project_dir else None)
    else:
        _recipe = recipe

    if not _recipe.is_multi_layer:
        raise ValueError(f"Recipe '{_recipe.name}' is not a multi-layer recipe")

    conn = _recipe.connection
    url = conn.get("url")
    version = conn.get("version", "2.0.0")
    if not url:
        raise ValueError(f"Recipe '{_recipe.name}': no url in connection")

    import xml.etree.ElementTree as ET
    import urllib.request

    caps_url = f"{url}?SERVICE=WFS&REQUEST=GetCapabilities&VERSION={version}"
    with urllib.request.urlopen(caps_url, timeout=30) as resp:
        tree = ET.parse(resp)
    root = tree.getroot()

    # Extract all FeatureType names (handle namespaces)
    remote_layers = set()
    for elem in root.iter():
        tag = elem.tag
        if tag.endswith("}Name") or tag == "Name":
            parent_tag = ""
            # Walk up to check parent is FeatureType
            # Simpler: just collect all Name elements under FeatureType
            pass
        if tag.endswith("}FeatureType") or tag == "FeatureType":
            for child in elem:
                if child.tag.endswith("}Name") or child.tag == "Name":
                    remote_layers.add(child.text.strip())
                    break

    recipe_wfs_layers = {
        cfg.get("wfs_layer", alias)
        for alias, cfg in _recipe.layers.items()
    }

    ok = sorted(recipe_wfs_layers & remote_layers)
    missing = sorted(recipe_wfs_layers - remote_layers)
    new = sorted(remote_layers - recipe_wfs_layers)

    return {"ok": ok, "missing": missing, "new": new}


def _run_osm_recipe(
    recipe: Recipe,
    *,
    input_boundary: "Path | str | None" = None,
    extent: tuple[float, float, float, float] | None = None,
    output_path: "Path | str | None" = None,
    recipe_dir: "Path | str | None" = None,
    crs: str | None = None,
    **kwargs,
) -> "gpd.GeoDataFrame":
    """Run an OSM recipe via download_osm_polygons."""
    from gis_utils.osm import bbox_from_shapefile, download_osm_polygons

    _crs = crs or recipe.connection.get("crs")
    if not _crs:
        raise ValueError(f"Recipe '{recipe.name}': crs is required (in recipe connection or as argument). No silent CRS defaults.")

    # Get bbox in WGS84
    if extent is not None:
        # extent is in project CRS, need WGS84 for OSM
        import geopandas as _gpd
        from shapely.geometry import box as _box
        _gdf = _gpd.GeoDataFrame(geometry=[_box(*extent)], crs=_crs)
        b = _gdf.to_crs("EPSG:4326").total_bounds
        bbox_wgs84 = tuple(b)
    elif input_boundary is not None:
        bbox_wgs84 = bbox_from_shapefile(input_boundary, crs="EPSG:4326")
    else:
        raise ValueError("OSM recipe requires input_boundary or extent")

    tags = recipe.connection.get("tags")
    if tags is None:
        tags = {
            "landuse": "^(residential|commercial|industrial|retail)$",
            "place": "^(city|town|village|hamlet|suburb|neighbourhood)$",
        }
    dissolve = recipe.detection.get("dissolve", True)

    # Check cache before calling to determine verbosity
    from gis_utils.osm import _osm_cache_key, CACHE_DIR_NAME as _OSM_CACHE_DIR
    _osm_cache_dir = Path.cwd() / _OSM_CACHE_DIR
    _osm_cache_file = _osm_cache_dir / _osm_cache_key(bbox_wgs84, tags, _crs, dissolve)
    _cache_hit = _osm_cache_file.exists()

    gdf = download_osm_polygons(bbox_wgs84, tags=tags, crs=_crs, dissolve=dissolve, **kwargs)

    # Apply exclude_tags filter: drop rows where specified columns match patterns
    if recipe.exclude_tags and len(gdf) > 0:
        import re
        drop_mask = None
        for col, pattern in recipe.exclude_tags.items():
            osm_col = col.replace(":", "_")
            if osm_col not in gdf.columns:
                continue
            col_match = gdf[osm_col].fillna("").astype(str).str.match(pattern)
            drop_mask = col_match if drop_mask is None else (drop_mask | col_match)
        if drop_mask is not None:
            n_before = len(gdf)
            gdf = gdf[~drop_mask].copy()
            n_dropped = n_before - len(gdf)
            if n_dropped > 0:
                print(f"  Excluded {n_dropped} features by exclude_tags filter")

    # Apply recipe post-processing pipeline
    _proj_dir = Path(recipe_dir) if recipe_dir else None

    if recipe.attribute_mappings:
        apply_attribute_mappings(gdf, recipe.attribute_mappings)
    if recipe.post_processing:
        gdf = apply_post_processing(gdf, recipe.post_processing)
    if recipe.hooks:
        gdf = load_and_run_hook(recipe.hooks, "post_process", gdf, _proj_dir)
    if recipe.column_mapping:
        is_shp = output_path is not None and str(output_path).lower().endswith(".shp")
        gdf = apply_column_mapping(gdf, recipe.column_mapping, is_shapefile=is_shp)

    # Write output
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext == ".shp":
            driver = "ESRI Shapefile"
        elif ext in (".gpkg", ".geopackage"):
            driver = "GPKG"
        elif ext == ".geojson":
            driver = "GeoJSON"
        else:
            driver = "GPKG"
        gdf.to_file(output_path, driver=driver)
        if not _cache_hit:
            print(f"[osm] Written: {output_path}", flush=True)

    return gdf

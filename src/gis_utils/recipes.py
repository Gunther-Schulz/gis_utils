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
    _source_path: Path | None = field(default=None, repr=False)


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
        recipes = [
            r for r in recipes
            if term in r.name.lower()
            or term in r.description.lower()
            or any(term in t.lower() for t in r.tags)
        ]

    return recipes


def resolve_connection(recipe: Recipe) -> dict[str, str]:
    """Resolve recipe connection to kwargs dict.

    Returns dict with keys depending on connection type:
    - WMS: {type, wms_url, wms_layer, crs}
    - WFS: {type, wfs_url, layer, crs, version}
    """
    conn = recipe.connection
    url = conn.get("url")
    conn_type = conn.get("type", "wms").lower()

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
    """Run a recipe, automatically dispatching to WFS or WMS based on connection type.

    This is the simplest way to use a recipe — just provide the recipe name
    and where you want the output.

    Args:
        recipe: Recipe name (string) or Recipe object.
        input_boundary: Shapefile/GeoPackage to derive extent from.
        extent: (minx, miny, maxx, maxy) bounding box.
        output_path: Output file (.gpkg recommended, .shp supported).
        recipe_dir: Project directory for recipe search.
        **kwargs: Additional arguments passed to the underlying WMS/WFS function.

    Returns:
        GeoDataFrame with the result.
    """
    if isinstance(recipe, str):
        _recipe = load_recipe(recipe, project_dir=Path(recipe_dir) if recipe_dir else None)
    else:
        _recipe = recipe

    conn = resolve_connection(_recipe)
    conn_type = conn.get("type", "wms")

    if conn_type == "wfs":
        from gis_utils.wfs import download
        return download(
            url=conn["wfs_url"],
            layer=conn.get("layer", ""),
            extent=extent,
            input_boundary=input_boundary,
            output_path=output_path,
            crs=conn.get("crs", "EPSG:25833"),
            version=conn.get("version", "1.1.0"),
            recipe=_recipe,
            recipe_dir=recipe_dir,
            **kwargs,
        )
    else:
        from gis_utils.wms import run
        return run(
            extent=extent,
            output_path=output_path,
            input_boundary=input_boundary,
            recipe=_recipe,
            recipe_dir=recipe_dir,
            **kwargs,
        )

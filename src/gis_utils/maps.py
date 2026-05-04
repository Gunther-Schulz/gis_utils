"""
Quick map rendering for GeoDataFrames.

Lightweight matplotlib + contextily wrapper for producing PNG overview maps
of one or more vector layers — typical use case: a quote/Angebot map showing
selected parcels on an OpenStreetMap basemap.

contextily is an optional dependency. If not installed, maps render without
a basemap (a warning is printed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt

LayerSpec = dict[str, Any] | gpd.GeoDataFrame


def _normalize_layers(
    layers: gpd.GeoDataFrame | Sequence[LayerSpec],
) -> list[dict[str, Any]]:
    """Coerce input into a list of layer-spec dicts."""
    if isinstance(layers, gpd.GeoDataFrame):
        return [{"gdf": layers}]
    out: list[dict[str, Any]] = []
    for item in layers:
        if isinstance(item, gpd.GeoDataFrame):
            out.append({"gdf": item})
        elif isinstance(item, dict):
            if "gdf" not in item:
                raise ValueError("layer dict must contain a 'gdf' key")
            out.append(item)
        else:
            raise TypeError(f"layer must be GeoDataFrame or dict, got {type(item)}")
    return out


def _resolve_provider(name: str | None):
    """Look up a contextily provider by dotted path (e.g. 'OpenStreetMap.Mapnik')."""
    import contextily as cx

    if name is None:
        return cx.providers.OpenStreetMap.Mapnik
    obj = cx.providers
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def quick_map(
    layers: gpd.GeoDataFrame | Sequence[LayerSpec],
    *,
    out_path: Path | str | None = None,
    title: str | None = None,
    label_field: str | None = None,
    label_size: float = 7,
    label_color: str = "black",
    figsize: tuple[float, float] = (12, 12),
    dpi: int = 200,
    basemap: bool = True,
    basemap_provider: str | None = None,
    extent: tuple[float, float, float, float] | gpd.GeoDataFrame | None = None,
    pad_factor: float = 0.4,
    legend: bool = True,
    axis_off: bool = True,
    return_fig: bool = False,
):
    """
    Render one or more GeoDataFrames as a PNG overview map.

    A layer may be a bare GeoDataFrame or a dict with rendering options:

        {"gdf": gdf, "facecolor": "#ff000055", "edgecolor": "red",
         "linewidth": 1.2, "linestyle": "--", "label": "Selection",
         "label_field": "name", "fit": True}

    Layer-level "label_field" overrides the function-level default.
    Layers are drawn in order — put context layers first, focus layers last.
    Set `fit=False` on a layer to exclude it from automatic extent
    calculation (useful for wide context layers like a whole Gemarkung).

    Args:
        layers: One GeoDataFrame, or a sequence of GeoDataFrames / layer dicts.
        out_path: Where to save the PNG. If None, the figure is not written.
        title: Plot title.
        label_field: Default attribute column to render as text at each
            feature centroid.
        label_size: Font size for labels.
        label_color: Font color for labels.
        figsize: matplotlib figure size in inches.
        dpi: Output resolution.
        basemap: If True, add an OSM basemap via contextily.
        basemap_provider: Dotted contextily provider path
            (e.g. "CartoDB.Positron"). Default OpenStreetMap.Mapnik.
        extent: (minx, miny, maxx, maxy) view extent in the data CRS,
            or a GeoDataFrame whose total_bounds will be used. If None, fit
            to layers with fit=True (or all layers if none have fit set),
            with `pad_factor` padding.
        pad_factor: Fractional padding applied when computing extent
            from layer bounds (0.4 = 40 % of the larger span).
        legend: If True, show legend for layers that have a "label".
        axis_off: Hide the matplotlib axis (recommended for clean overviews).
        return_fig: If True, return the (fig, ax) tuple instead of closing.

    Returns:
        Path to the saved PNG (or None if out_path is None and not return_fig),
        or (fig, ax) when return_fig=True.

    Raises:
        ValueError: if no layers provided or layers have inconsistent CRS.
    """
    specs = _normalize_layers(layers)
    if not specs:
        raise ValueError("at least one layer is required")

    crs_set = {s["gdf"].crs for s in specs if s["gdf"].crs is not None}
    if len(crs_set) > 1:
        raise ValueError(f"layers must share a CRS, got {crs_set}")
    crs = next(iter(crs_set), None)

    fig, ax = plt.subplots(figsize=figsize)

    legend_handles: list[Any] = []
    for spec in specs:
        gdf = spec["gdf"]
        if gdf.empty:
            continue
        plot_kwargs = {
            k: spec[k]
            for k in (
                "facecolor", "edgecolor", "linewidth", "linestyle",
                "alpha", "color", "marker", "markersize", "hatch",
            )
            if k in spec
        }
        plot_kwargs.setdefault("facecolor", "#0066ff44")
        plot_kwargs.setdefault("edgecolor", "#0033aa")
        plot_kwargs.setdefault("linewidth", 1.0)
        gdf.plot(ax=ax, **plot_kwargs)

        if "label" in spec:
            from matplotlib.patches import Patch
            legend_handles.append(Patch(
                facecolor=plot_kwargs.get("facecolor", "none"),
                edgecolor=plot_kwargs.get("edgecolor", "black"),
                linestyle=plot_kwargs.get("linestyle", "-"),
                linewidth=plot_kwargs.get("linewidth", 1.0),
                label=spec["label"],
            ))

        lbl_field = spec.get("label_field", label_field)
        if lbl_field and lbl_field in gdf.columns:
            for _, row in gdf.iterrows():
                c = row.geometry.centroid
                ax.annotate(
                    str(row[lbl_field]),
                    (c.x, c.y),
                    ha="center", va="center",
                    fontsize=label_size, color=label_color,
                )

    # View extent
    if extent is not None:
        if isinstance(extent, gpd.GeoDataFrame):
            minx, miny, maxx, maxy = extent.total_bounds
            pad = max(maxx - minx, maxy - miny) * pad_factor
            minx -= pad; miny -= pad; maxx += pad; maxy += pad
        else:
            minx, miny, maxx, maxy = extent
    else:
        # Use only layers with fit=True; if no layer specifies it, use all
        any_fit = any("fit" in s for s in specs)
        fit_specs = [
            s for s in specs
            if not s["gdf"].empty and (s.get("fit", True) if any_fit else True)
        ]
        if not fit_specs:
            fit_specs = [s for s in specs if not s["gdf"].empty]
        if not fit_specs:
            raise ValueError("all layers are empty")
        bounds = [s["gdf"].total_bounds for s in fit_specs]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        pad = max(maxx - minx, maxy - miny) * pad_factor
        minx -= pad; miny -= pad; maxx += pad; maxy += pad
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    if basemap:
        try:
            import contextily as cx
            cx.add_basemap(
                ax,
                crs=crs,
                source=_resolve_provider(basemap_provider),
            )
        except ImportError:
            print("[quick_map] contextily not installed — skipping basemap")
        except Exception as exc:
            print(f"[quick_map] basemap failed: {exc}")

    if title:
        ax.set_title(title)
    if legend and legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    if axis_off:
        ax.set_axis_off()
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if return_fig:
        return fig, ax
    plt.close(fig)
    return out_path

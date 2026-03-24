"""
Extract geometry from DXF files into GeoDataFrames.

Handles all common entity types (LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE,
ELLIPSE, HATCH, POINT, TEXT, MTEXT) with proper bulge/arc interpolation
and recursive block (INSERT) processing with affine transforms.

Consolidates best approaches from multiple project scripts:
- Schwerin: ezdxf.math helpers for bulge, clean ring closure
- Winnert extract_layers_to_shp_fixed: adaptive arc segmentation, full entity coverage
- Winnert extract_everything: entity-to-LWPOLYLINE, HATCH edge handling
- K36 extract_circles_pappeln: circle-specific extraction with transforms
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.math import bulge_center, bulge_radius
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _p2(p) -> tuple[float, float]:
    """Extract 2D coords from any ezdxf point-like object."""
    return (float(p[0]), float(p[1]))


def _arc_segment_count(theta: float, min_segments: int = 16) -> int:
    """Adaptive segment count based on arc angle. More segments for larger arcs."""
    return max(min_segments, int(abs(theta) * 32 / math.pi))


def interpolate_bulge_arc(
    start: tuple[float, float],
    end: tuple[float, float],
    bulge: float,
    num_points: int = 0,
) -> list[tuple[float, float]]:
    """
    Compute intermediate arc points between two vertices connected by a DXF bulge.

    Uses ezdxf.math.bulge_center/bulge_radius for robust center/radius calculation,
    then interpolates intermediate points along the arc.

    Args:
        start: Start vertex (x, y).
        end: End vertex (x, y).
        bulge: DXF bulge value. 0 = straight, positive = CCW, negative = CW.
        num_points: Number of intermediate points. 0 = auto-scale with arc size.

    Returns:
        List of intermediate (x, y) points (excludes start and end vertices).
        Empty list if bulge is effectively zero.
    """
    if abs(bulge) < 1e-12:
        return []

    try:
        center = bulge_center(start, end, bulge)
        radius = bulge_radius(start, end, bulge)
    except (ZeroDivisionError, ValueError):
        return []

    cx, cy = float(center.x), float(center.y)
    sa = math.atan2(start[1] - cy, start[0] - cx)
    span = 4.0 * math.atan(bulge)  # signed: positive=CCW, negative=CW

    if num_points <= 0:
        num_points = _arc_segment_count(span)

    return [
        (cx + radius * math.cos(sa + (k / num_points) * span),
         cy + radius * math.sin(sa + (k / num_points) * span))
        for k in range(1, num_points)
    ]


def lwpolyline_to_coords(
    entity,
    arc_points: int = 0,
) -> list[tuple[float, float]]:
    """
    Extract coordinate list from an LWPOLYLINE or 2D POLYLINE, interpolating bulge arcs.

    Args:
        entity: An ezdxf LWPOLYLINE or POLYLINE entity.
        arc_points: Points per arc segment. 0 = adaptive.

    Returns:
        List of (x, y) tuples. For closed entities, the last point equals the first.
    """
    entity_type = entity.dxftype()

    if entity_type == "LWPOLYLINE":
        raw = list(entity.get_points("xyb"))
        vertices = [_p2((p[0], p[1])) for p in raw]
        bulges = [float(p[2]) if len(p) > 2 else 0.0 for p in raw]
        is_closed = entity.closed
    elif entity_type == "POLYLINE":
        verts = list(entity.vertices)
        vertices = [_p2(v.dxf.location) for v in verts]
        bulges = [float(v.dxf.get("bulge", 0)) for v in verts]
        is_closed = entity.is_closed
    else:
        return []

    n = len(vertices)
    if n < 2:
        return list(vertices)

    out: list[tuple[float, float]] = [vertices[0]]
    loop_end = n if is_closed else n - 1

    for i in range(loop_end):
        s = vertices[i]
        e = vertices[(i + 1) % n]
        b = bulges[i]
        arc_pts = interpolate_bulge_arc(s, e, b, arc_points)
        out.extend(arc_pts)
        out.append(e)

    # Ensure exact closure for closed entities
    if is_closed and len(out) >= 3:
        if abs(out[-1][0] - out[0][0]) > 1e-9 or abs(out[-1][1] - out[0][1]) > 1e-9:
            out.append(out[0])
        else:
            out[-1] = out[0]  # exact match

    return out


def _arc_to_coords(entity, min_segments: int = 16) -> list[tuple[float, float]]:
    """Convert a DXF ARC entity to a list of points."""
    center = entity.dxf.center
    radius = entity.dxf.radius
    start_deg = entity.dxf.start_angle
    end_deg = entity.dxf.end_angle

    angle_range = end_deg - start_deg
    if angle_range < 0:
        angle_range += 360

    num_segments = max(min_segments, int(angle_range / 360 * 64))
    return [
        (center[0] + radius * math.cos(math.radians(start_deg + angle_range * i / num_segments)),
         center[1] + radius * math.sin(math.radians(start_deg + angle_range * i / num_segments)))
        for i in range(num_segments + 1)
    ]


def _circle_to_coords(entity, num_segments: int = 64) -> list[tuple[float, float]]:
    """Convert a DXF CIRCLE entity to a closed polygon coordinate list."""
    cx, cy = entity.dxf.center[0], entity.dxf.center[1]
    r = entity.dxf.radius
    pts = [
        (cx + r * math.cos(2 * math.pi * i / num_segments),
         cy + r * math.sin(2 * math.pi * i / num_segments))
        for i in range(num_segments)
    ]
    pts.append(pts[0])
    return pts


def _ellipse_to_coords(entity, num_segments: int = 64) -> list[tuple[float, float]]:
    """Convert a DXF ELLIPSE entity to a coordinate list."""
    center = entity.dxf.center
    major_axis = entity.dxf.major_axis
    ratio = entity.dxf.ratio
    start_param = entity.dxf.get("start_param", 0)
    end_param = entity.dxf.get("end_param", 2 * math.pi)

    major_len = math.sqrt(major_axis[0] ** 2 + major_axis[1] ** 2)
    major_angle = math.atan2(major_axis[1], major_axis[0])
    minor_len = major_len * ratio
    cos_a, sin_a = math.cos(major_angle), math.sin(major_angle)

    pts = []
    for i in range(num_segments):
        param = start_param + (end_param - start_param) * i / num_segments
        xl = major_len * math.cos(param)
        yl = minor_len * math.sin(param)
        pts.append((center[0] + xl * cos_a - yl * sin_a,
                     center[1] + xl * sin_a + yl * cos_a))

    if (end_param - start_param) >= (2 * math.pi - 0.01):
        pts.append(pts[0])
    return pts


def _hatch_to_coords(entity) -> list[tuple[float, float]] | None:
    """Extract first boundary path from a HATCH entity as coordinates."""
    for path in entity.paths:
        vertices: list[tuple[float, float]] = []
        if hasattr(path, "vertices"):
            vertices = [_p2(v) for v in path.vertices]
        elif hasattr(path, "edges"):
            for edge in path.edges:
                edge_type = type(edge).__name__
                if edge_type == "LineEdge":
                    vertices.append(_p2(edge.start))
                elif edge_type == "ArcEdge":
                    # Approximate arc edge
                    cx, cy = edge.center
                    r = edge.radius
                    sa = math.radians(edge.start_angle)
                    ea = math.radians(edge.end_angle)
                    if edge.ccw:
                        if ea <= sa:
                            ea += 2 * math.pi
                    else:
                        if ea >= sa:
                            ea -= 2 * math.pi
                    n = max(8, int(abs(ea - sa) * 16 / math.pi))
                    for k in range(n + 1):
                        a = sa + (ea - sa) * k / n
                        vertices.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            if vertices and hasattr(path.edges[-1], "end"):
                vertices.append(_p2(path.edges[-1].end))
        if len(vertices) >= 3:
            return vertices
    return None


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_entity(
    entity,
    layer_name: str | None = None,
    circles_as_points: bool = False,
    arc_points: int = 0,
) -> tuple[list[tuple[float, float]] | None, str, dict[str, Any]]:
    """
    Extract geometry from a single DXF entity.

    Returns:
        (coords, geom_type, extra_attrs) or (None, "", {}) on failure.
        geom_type is one of: "Point", "LineString", "Polygon", "Hatch".
    """
    entity_type = entity.dxftype()
    extra: dict[str, Any] = {}

    try:
        if entity_type == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            return [_p2(s), _p2(e)], "LineString", extra

        if entity_type in ("LWPOLYLINE", "POLYLINE"):
            coords = lwpolyline_to_coords(entity, arc_points)
            if not coords:
                return None, "", extra
            is_closed = (entity.closed if entity_type == "LWPOLYLINE"
                         else entity.is_closed)
            gtype = "Polygon" if is_closed and len(coords) >= 4 else "LineString"
            return coords, gtype, extra

        if entity_type == "ARC":
            return _arc_to_coords(entity), "LineString", extra

        if entity_type == "CIRCLE":
            center = entity.dxf.center
            radius = entity.dxf.radius
            if circles_as_points:
                extra["radius"] = radius
                return [_p2(center)], "Point", extra
            return _circle_to_coords(entity), "Polygon", extra

        if entity_type == "ELLIPSE":
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            coords = _ellipse_to_coords(entity)
            if circles_as_points:
                major_len = math.sqrt(major_axis[0] ** 2 + major_axis[1] ** 2)
                extra["major_axis"] = major_len
                extra["minor_axis"] = major_len * ratio
                return [_p2(center)], "Point", extra
            is_closed = len(coords) > 0 and coords[-1] == coords[0]
            return coords, "Polygon" if is_closed else "LineString", extra

        if entity_type == "HATCH":
            coords = _hatch_to_coords(entity)
            if coords:
                extra["is_hatch"] = True
                return coords, "Hatch", extra
            return None, "", extra

        if entity_type == "POINT":
            return [_p2(entity.dxf.location)], "Point", extra

        if entity_type in ("TEXT", "MTEXT"):
            return [_p2(entity.dxf.insert)], "Point", extra

    except Exception:
        pass

    return None, "", extra


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _apply_transform(
    points: list[tuple[float, float]],
    matrix: tuple[float, float, float, float, float],
) -> list[tuple[float, float]]:
    """Apply 2D affine transform: scale → rotate → translate."""
    x_off, y_off, rot, sx, sy = matrix
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    out = []
    for x, y in points:
        xs, ys = x * sx, y * sy
        out.append((xs * cos_r - ys * sin_r + x_off,
                     xs * sin_r + ys * cos_r + y_off))
    return out


def _compose_transform(
    parent: tuple[float, float, float, float, float],
    insert_point: tuple[float, float, float],
    rotation_deg: float,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float, float, float, float]:
    """Compose parent transform with an INSERT's local transform."""
    px, py, prot, psx, psy = parent
    cos_p, sin_p = math.cos(prot), math.sin(prot)
    ix, iy = insert_point[0], insert_point[1]
    tx = ix * psx * cos_p - iy * psy * sin_p + px
    ty = ix * psx * sin_p + iy * psy * cos_p + py
    return (tx, ty, prot + math.radians(rotation_deg), psx * x_scale, psy * y_scale)


IDENTITY_TRANSFORM = (0.0, 0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Recursive block processing
# ---------------------------------------------------------------------------

def _process_block_recursive(
    entity,
    entity_layer: str,
    transform: tuple[float, float, float, float, float],
    doc: ezdxf.document.Drawing,
    *,
    exclude_layers: set[str],
    circles_as_points: bool,
    arc_points: int,
    depth: int = 0,
    max_depth: int = 10,
) -> list[tuple[str, dict[str, Any]]]:
    """Recursively extract geometry from an entity, handling nested INSERTs."""
    if depth > max_depth:
        return []

    if entity.dxftype() == "INSERT":
        block_name = entity.dxf.name
        if block_name not in doc.blocks:
            return []
        block = doc.blocks.get(block_name)

        child_transform = _compose_transform(
            transform,
            entity.dxf.insert,
            entity.dxf.get("rotation", 0),
            entity.dxf.get("xscale", 1),
            entity.dxf.get("yscale", 1),
        )

        features = []
        for child in block:
            child_layer = getattr(child.dxf, "layer", entity_layer)
            if child_layer in exclude_layers:
                continue
            features.extend(_process_block_recursive(
                child, child_layer, child_transform, doc,
                exclude_layers=exclude_layers,
                circles_as_points=circles_as_points,
                arc_points=arc_points,
                depth=depth + 1,
                max_depth=max_depth,
            ))
        return features

    # Regular entity — extract and transform
    coords, gtype, extra = _extract_entity(
        entity, entity_layer,
        circles_as_points=circles_as_points,
        arc_points=arc_points,
    )
    if not coords:
        return []

    transformed = _apply_transform(coords, transform)

    # Scale dimension attributes
    _, _, _, sx, sy = transform
    avg_scale = (abs(sx) + abs(sy)) / 2
    for key in ("radius", "major_axis", "minor_axis"):
        if key in extra:
            extra[key] *= avg_scale

    return [(entity_layer, _build_feature(transformed, gtype, entity.dxftype(), extra))]


def _build_feature(
    coords: list[tuple[float, float]],
    geom_type: str,
    entity_type: str,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a feature dict with Shapely geometry from coords and type."""
    try:
        if geom_type == "Point" and len(coords) == 1:
            geom = Point(coords[0])
        elif geom_type in ("Polygon", "Hatch") and len(coords) >= 3:
            geom = Polygon(coords)
            if not geom.is_valid:
                geom = make_valid(geom)
                if geom.is_empty:
                    return None
                # If make_valid produced a MultiPolygon, take the largest part
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: g.area)
        elif geom_type == "LineString" and len(coords) >= 2:
            geom = LineString(coords)
        else:
            return None

        if geom.is_empty:
            return None

        feature: dict[str, Any] = {
            "geometry": geom,
            "entity_type": entity_type,
        }
        feature.update(extra)
        return feature
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_dxf_layers(
    dxf_path: str | Path,
    crs: str,
    *,
    layers: list[str] | None = None,
    exclude_layers: list[str] | None = None,
    arc_points: int = 0,
    circles_as_points: bool = False,
    process_blocks: bool = True,
    max_block_depth: int = 10,
) -> dict[str, dict[str, gpd.GeoDataFrame]]:
    """
    Extract all geometry from a DXF file, organized by layer and geometry type.

    Args:
        dxf_path: Path to DXF file.
        crs: Coordinate reference system string (e.g. "EPSG:25833").
        layers: Only extract these layers. None = all layers.
        exclude_layers: Skip these layers.
        arc_points: Points per arc/bulge interpolation. 0 = adaptive.
        circles_as_points: If True, extract circles/ellipses as center Points
            with radius attributes instead of polygon approximations.
        process_blocks: If True, recursively extract geometry from INSERT
            (block reference) entities with proper affine transforms.
        max_block_depth: Maximum recursion depth for nested blocks.

    Returns:
        Nested dict: ``{layer_name: {geom_type: GeoDataFrame}}``.
        geom_type keys: ``"Point"``, ``"LineString"``, ``"Polygon"``, ``"Hatch"``.
        Each GeoDataFrame has columns: geometry, entity_type, plus any extras
        (radius, major_axis, minor_axis for circles/ellipses as points).
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    exclude = set(exclude_layers or [])
    include = set(layers) if layers else None

    # Collect features: {layer: [feature_dict, ...]}
    layer_features: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # 1. Process modelspace entities directly
    for entity in msp:
        layer = getattr(entity.dxf, "layer", None)
        if layer is None:
            continue
        if layer in exclude:
            continue
        if include and layer not in include:
            continue

        if entity.dxftype() == "INSERT" and process_blocks:
            results = _process_block_recursive(
                entity, layer, IDENTITY_TRANSFORM, doc,
                exclude_layers=exclude,
                circles_as_points=circles_as_points,
                arc_points=arc_points,
                max_depth=max_block_depth,
            )
            for feat_layer, feat in results:
                if feat is not None:
                    if include and feat_layer not in include:
                        continue
                    layer_features[feat_layer].append(feat)
        else:
            coords, gtype, extra = _extract_entity(
                entity, layer,
                circles_as_points=circles_as_points,
                arc_points=arc_points,
            )
            if coords:
                feat = _build_feature(coords, gtype, entity.dxftype(), extra)
                if feat is not None:
                    layer_features[layer].append(feat)

    # 2. Organize into {layer: {geom_type: GeoDataFrame}}
    result: dict[str, dict[str, gpd.GeoDataFrame]] = {}

    for layer, features in layer_features.items():
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for feat in features:
            geom = feat["geometry"]
            if geom.geom_type == "Point":
                by_type["Point"].append(feat)
            elif geom.geom_type == "LineString":
                by_type["LineString"].append(feat)
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                key = "Hatch" if feat.get("is_hatch") else "Polygon"
                by_type[key].append(feat)

        layer_gdfs: dict[str, gpd.GeoDataFrame] = {}
        for gtype, feats in by_type.items():
            gdf = gpd.GeoDataFrame(feats, crs=crs)
            gdf.drop(columns=["is_hatch"], errors="ignore", inplace=True)
            layer_gdfs[gtype] = gdf
        result[layer] = layer_gdfs

    return result


def extract_dxf_circles(
    dxf_path: str | Path,
    crs: str,
    *,
    layers: list[str] | None = None,
    process_blocks: bool = True,
    max_block_depth: int = 10,
) -> gpd.GeoDataFrame:
    """
    Extract circle centers as a Point GeoDataFrame with radius attribute.

    Convenience wrapper around extract_dxf_layers() for circle-specific extraction.

    Args:
        dxf_path: Path to DXF file.
        crs: Coordinate reference system string.
        layers: Only extract circles from these layers. None = all.
        process_blocks: Recurse into block references.
        max_block_depth: Maximum block nesting depth.

    Returns:
        GeoDataFrame with Point geometries and a ``radius`` column.
    """
    all_layers = extract_dxf_layers(
        dxf_path, crs,
        layers=layers,
        circles_as_points=True,
        process_blocks=process_blocks,
        max_block_depth=max_block_depth,
    )

    frames = []
    for layer_name, gdfs in all_layers.items():
        if "Point" in gdfs:
            gdf = gdfs["Point"].copy()
            if "radius" in gdf.columns:
                gdf["layer"] = layer_name
                frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["geometry", "radius", "layer"], crs=crs)

    return gpd.GeoDataFrame(
        gpd.pd.concat(frames, ignore_index=True), crs=crs
    )


def save_layers_as_shapefiles(
    layers: dict[str, dict[str, gpd.GeoDataFrame]],
    output_dir: str | Path,
) -> list[Path]:
    """
    Write extract_dxf_layers() output to organized shapefiles.

    Creates subdirectories per geometry type (Point/, LineString/, Polygon/, Hatch/).

    Args:
        layers: Output from extract_dxf_layers().
        output_dir: Root output directory.

    Returns:
        List of written shapefile paths.
    """
    import re

    output_dir = Path(output_dir)
    written: list[Path] = []

    for layer_name, gdfs in layers.items():
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", layer_name).strip()
        if not safe_name:
            safe_name = "unnamed"

        for gtype, gdf in gdfs.items():
            if gdf.empty:
                continue
            subdir = output_dir / gtype
            subdir.mkdir(parents=True, exist_ok=True)
            path = subdir / f"{safe_name}.shp"
            gdf.to_file(path, driver="ESRI Shapefile")
            written.append(path)

    return written

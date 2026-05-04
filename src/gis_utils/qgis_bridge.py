"""Optional bridge to a running QGIS instance via the qgis-mcp plugin.

This module is **optional** — gis_utils stays headless-capable.  If the
optional ``[qgis]`` extra is not installed (or the ``qgis-mcp`` Python
package is otherwise unavailable), every public function in this module
becomes a silent no-op.  This lets workflow runners and templates
**opt in** to live-QGIS interaction without breaking on machines / CI
runs without QGIS.

Activation requires three things:

1. ``pip install gis-utils[qgis]`` (or ``pip install qgis-mcp`` directly).
2. The matching QGIS plugin (``QGIS MCP`` by N. Karasiak) installed and
   enabled in QGIS, with its socket server started
   (Plugins → QGIS MCP → Start Server, port 9876 by default).
3. A QGIS process actually running with that plugin active.

If any of those is missing, :func:`is_available` returns ``False`` and
all other helpers print a single warning and return without action.

Typical use cases
-----------------

- **Auto-reload after workflow step**: ``reload_paths(['Shape/Layer.shp'])``
  refreshes any layer in the open project whose data source matches.
- **Open generated layer**: ``add_layer('Shape/Result.gpkg')`` adds it to
  the current project (idempotent — won't add twice).
- **Custom code**: ``execute('iface.mapCanvas().refresh()')`` for ad-hoc
  PyQGIS calls.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Iterable

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876
PROBE_TIMEOUT_S = 0.3


def _import_client():
    """Lazy import of qgis_mcp.client.  Returns class or None."""
    try:
        from qgis_mcp.client import QgisMCPClient  # type: ignore[import-not-found]
        return QgisMCPClient
    except ImportError:
        return None


def is_available(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Cheap reachability probe — does the QGIS plugin's socket server respond?

    Performs a quick TCP connect with a short timeout.  Does NOT load the
    qgis-mcp Python package.  Returns False if either the package or the
    plugin server is missing — both situations make all bridge calls
    silently no-op.
    """
    if _import_client() is None:
        return False
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
            return True
    except (OSError, socket.timeout):
        return False


def _connect(host: str, port: int):
    """Open a short-lived client connection.  Returns None if unavailable."""
    cls = _import_client()
    if cls is None:
        return None
    try:
        client = cls(host=host, port=port)
        client.connect()
        return client
    except Exception as exc:
        print(f"[qgis_bridge] Cannot connect to QGIS at {host}:{port}: {exc}")
        return None


def reload_paths(
    paths: Iterable[str | Path],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> int:
    """Reload all layers in the open QGIS project whose source matches one
    of *paths*.

    Source-path matching uses ``layer.source().startswith(absolute_path)``
    so a single GeoPackage file matches all its layers, and shapefiles
    match by their absolute path prefix.

    Args:
        paths: Iterable of file paths (relative or absolute).  Relative
            paths are resolved against the current working directory.

    Returns:
        Number of layers reloaded.  ``0`` if QGIS is not reachable or
        no layers matched.
    """
    abs_paths = [str(Path(p).resolve()) for p in paths]
    if not abs_paths:
        return 0

    client = _connect(host, port)
    if client is None:
        return 0

    code = (
        "from qgis.core import QgsProject\n"
        f"_paths = {abs_paths!r}\n"
        "_n = 0\n"
        "for _l in QgsProject.instance().mapLayers().values():\n"
        "    _src = _l.source()\n"
        "    if any(_src.startswith(_p) for _p in _paths):\n"
        "        _l.dataProvider().reloadData()\n"
        "        _l.triggerRepaint()\n"
        "        _n += 1\n"
        "iface.mapCanvas().refresh() if _n else None\n"
        "result = _n\n"
    )
    try:
        resp = client.execute_code(code)
        # client returns dict {'status': 'success', 'result': {...}}
        n = 0
        if isinstance(resp, dict):
            inner = resp.get("result")
            if isinstance(inner, dict):
                n = int(inner.get("result", 0) or 0)
        if n:
            print(f"[qgis_bridge] Reloaded {n} layer(s) in QGIS")
        return n
    except Exception as exc:
        print(f"[qgis_bridge] reload_paths failed: {exc}")
        return 0
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


_RASTER_EXTENSIONS = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg",
                      ".gif", ".bmp", ".vrt", ".asc", ".img", ".dem"}


def _is_raster_path(path: Path) -> bool:
    """Heuristic — is this a raster file based on extension?"""
    return path.suffix.lower() in _RASTER_EXTENSIONS


def add_layer(
    path: str | Path,
    *,
    name: str | None = None,
    provider: str = "ogr",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> bool:
    """Add a vector layer to the open QGIS project (idempotent).

    If a layer with the same source path already exists it is reloaded
    instead of being added a second time.  Returns ``True`` on success
    (layer added or reloaded), ``False`` if QGIS not reachable.
    """
    p = Path(path).resolve()
    client = _connect(host, port)
    if client is None:
        return False
    try:
        # Check if already in project — reload instead of double-adding.
        existing = client.execute_code(
            f"from qgis.core import QgsProject\n"
            f"result = any(l.source().startswith({str(p)!r}) "
            f"for l in QgsProject.instance().mapLayers().values())\n"
        )
        already = False
        if isinstance(existing, dict):
            inner = existing.get("result")
            if isinstance(inner, dict):
                already = bool(inner.get("result"))
        if already:
            return reload_paths([p], host=host, port=port) > 0
        client.add_vector_layer(str(p), name=name, provider=provider)
        return True
    except Exception as exc:
        print(f"[qgis_bridge] add_layer failed for {p}: {exc}")
        return False
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def add_raster(
    path: str | Path,
    *,
    name: str | None = None,
    provider: str = "gdal",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> bool:
    """Add a raster layer to the open QGIS project (idempotent).

    If a layer with the same source path already exists it is reloaded
    instead of being added a second time.  Returns ``True`` on success.
    """
    p = Path(path).resolve()
    client = _connect(host, port)
    if client is None:
        return False
    try:
        existing = client.execute_code(
            f"from qgis.core import QgsProject\n"
            f"result = any(l.source().startswith({str(p)!r}) "
            f"for l in QgsProject.instance().mapLayers().values())\n"
        )
        already = False
        if isinstance(existing, dict):
            inner = existing.get("result")
            if isinstance(inner, dict):
                already = bool(inner.get("result"))
        if already:
            return reload_paths([p], host=host, port=port) > 0
        client.add_raster_layer(str(p), name=name, provider=provider)
        return True
    except Exception as exc:
        print(f"[qgis_bridge] add_raster failed for {p}: {exc}")
        return False
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def open_path(
    path: str | Path,
    *,
    name: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> bool:
    """Auto-detect raster vs vector and add the appropriate layer type.

    Detection by file extension; vector is the default fallback so that
    GeoPackage / Shapefile / GeoJSON / DXF / etc. are correctly added.
    """
    p = Path(path)
    if _is_raster_path(p):
        return add_raster(p, name=name, host=host, port=port)
    return add_layer(p, name=name, host=host, port=port)


def apply_qml(
    layer_path: str | Path,
    qml_path: str | Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> int:
    """Apply a QML style to layer(s) in the open QGIS project whose source
    matches ``layer_path``.

    Returns number of layers styled.  ``0`` if QGIS not reachable, no
    matching layer in project, or QML file missing.  No-op if file
    doesn't exist (logged warning, not error).
    """
    lp = Path(layer_path).resolve()
    qp = Path(qml_path).resolve()
    if not qp.is_file():
        print(f"[qgis_bridge] QML not found: {qp}")
        return 0
    client = _connect(host, port)
    if client is None:
        return 0
    code = (
        "from qgis.core import QgsProject\n"
        f"_lp = {str(lp)!r}\n"
        f"_qml = {str(qp)!r}\n"
        "_n = 0\n"
        "for _l in QgsProject.instance().mapLayers().values():\n"
        "    if _l.source().startswith(_lp):\n"
        "        _ok, _msg = _l.loadNamedStyle(_qml)\n"
        "        if _ok:\n"
        "            _l.triggerRepaint()\n"
        "            _n += 1\n"
        "iface.mapCanvas().refresh() if _n else None\n"
        "result = _n\n"
    )
    try:
        resp = client.execute_code(code)
        n = 0
        if isinstance(resp, dict):
            inner = resp.get("result")
            if isinstance(inner, dict):
                n = int(inner.get("result", 0) or 0)
        if n:
            print(f"[qgis_bridge] Applied QML {qp.name} to {n} layer(s)")
        return n
    except Exception as exc:
        print(f"[qgis_bridge] apply_qml failed: {exc}")
        return 0
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def take_canvas_screenshot(
    out_path: str | Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> bool:
    """Save the current QGIS map canvas to a PNG file.

    Returns True on success, False if QGIS unreachable.  Uses PyQGIS
    canvas grab — fast, no re-render, captures whatever the user is
    looking at right now.
    """
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    client = _connect(host, port)
    if client is None:
        return False
    code = (
        f"_out = {str(out)!r}\n"
        "_canvas = iface.mapCanvas()\n"
        "_pixmap = _canvas.grab()\n"
        "_pixmap.save(_out)\n"
        "result = _out\n"
    )
    try:
        client.execute_code(code)
        return out.is_file()
    except Exception as exc:
        print(f"[qgis_bridge] take_canvas_screenshot failed: {exc}")
        return False
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def execute(
    code: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
):
    """Execute arbitrary PyQGIS code in the running QGIS instance.

    Returns the value of the ``result`` variable set inside *code*, or
    ``None`` if QGIS is not reachable.
    """
    client = _connect(host, port)
    if client is None:
        return None
    try:
        return client.execute_code(code)
    except Exception as exc:
        print(f"[qgis_bridge] execute failed: {exc}")
        return None
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def auto_reload_enabled() -> bool:
    """Whether the workflow runner should auto-reload after each step.

    Controlled by environment variable ``GIS_WORKFLOW_QGIS_RELOAD``.
    """
    return _env_truthy("GIS_WORKFLOW_QGIS_RELOAD")


def auto_open_enabled() -> bool:
    """Whether the workflow runner should auto-open new outputs in QGIS.

    Controlled by environment variable ``GIS_WORKFLOW_QGIS_OPEN``.
    Independent of auto_reload — open=add new, reload=refresh existing.
    Both can be active at once (recommended for live development).
    """
    return _env_truthy("GIS_WORKFLOW_QGIS_OPEN")


def screenshots_dir() -> Path | None:
    """Directory for the workflow runner's auto-screenshot audit trail.

    Set via env var ``GIS_WORKFLOW_QGIS_SCREENSHOTS`` to a directory path.
    Returns Path or None when disabled / not set.
    """
    val = os.environ.get("GIS_WORKFLOW_QGIS_SCREENSHOTS", "").strip()
    return Path(val) if val else None

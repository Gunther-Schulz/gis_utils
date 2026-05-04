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


def auto_reload_enabled() -> bool:
    """Whether the workflow runner should auto-reload after each step.

    Controlled by environment variable ``GIS_WORKFLOW_QGIS_RELOAD``.
    Truthy values (``1``, ``true``, ``yes``, ``on``, case-insensitive)
    enable it; everything else (including unset) disables it.
    """
    val = os.environ.get("GIS_WORKFLOW_QGIS_RELOAD", "").strip().lower()
    return val in {"1", "true", "yes", "on"}

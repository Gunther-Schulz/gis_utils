"""Internal helper to read WMS/WFS/WCS connections from the QGIS config file.

Not part of the public API — used by recipes.py to resolve qgis_name references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote


@dataclass
class QgisConnection:
    name: str
    service_type: str  # "wms" | "wfs" | "wcs"
    url: str = ""
    username: str = ""
    password: str = ""
    raw: dict[str, str] = field(default_factory=dict)


_DEFAULT_CONFIG = (
    Path.home()
    / ".local/share/QGIS/QGIS3/profiles/default/QGIS/QGIS3.ini"
)

# Pattern: ows\items\{type}\connections\items\{encoded_name}\{field}={value}
_LINE_RE = re.compile(
    r"^ows\\items\\(wms|wfs|wcs)\\connections\\items\\([^\\]+)\\([^=]+)=(.*)$"
)


def _parse_connections(config_path: Path) -> list[QgisConnection]:
    if not config_path.is_file():
        return []

    grouped: dict[tuple[str, str], dict[str, str]] = {}
    in_connections = False

    for raw_line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_connections = line == "[connections]"
            continue
        if not in_connections:
            continue

        m = _LINE_RE.match(line)
        if not m:
            continue

        svc_type, encoded_name, field_name, value = m.groups()
        # Strip surrounding quotes if present
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]

        key = (svc_type, encoded_name)
        grouped.setdefault(key, {})[field_name] = value

    results = []
    for (svc_type, encoded_name), fields in grouped.items():
        decoded_name = unquote(encoded_name, encoding="latin-1")
        conn = QgisConnection(
            name=decoded_name,
            service_type=svc_type,
            url=fields.get("url", ""),
            username=fields.get("username", ""),
            password=fields.get("password", ""),
            raw=fields,
        )
        results.append(conn)

    results.sort(key=lambda c: (c.service_type, c.name.lower()))
    return results


def qgis_connections(
    service_type: str | None = None,
    config_path: Path | None = None,
) -> list[QgisConnection]:
    """List QGIS OWS connections, optionally filtered by service type."""
    path = config_path or _DEFAULT_CONFIG
    conns = _parse_connections(path)
    if service_type:
        conns = [c for c in conns if c.service_type == service_type.lower()]
    return conns


def qgis_connection(
    name: str,
    service_type: str | None = None,
    config_path: Path | None = None,
) -> QgisConnection | None:
    """Get a single connection by name (case-insensitive)."""
    for c in qgis_connections(service_type, config_path):
        if c.name.lower() == name.lower():
            return c
    return None

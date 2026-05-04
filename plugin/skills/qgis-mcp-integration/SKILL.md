---
name: qgis-mcp-integration
description: This skill should be used when the user asks to "set up QGIS MCP", "QGIS-Live-Bridge", "auto-reload layers in QGIS", "render map from QGIS", "open generated layer in QGIS", "qgis_bridge", "GIS_WORKFLOW_QGIS_RELOAD", "QGIS plugin installieren", "Claude soll QGIS steuern", "MCP server QGIS", or any task involving the live integration between gis_utils workflows and a running QGIS instance via the qgis-mcp plugin.
license: MIT
---

## QGIS-MCP Integration with gis_utils

The optional **`gis_utils.qgis_bridge`** module connects gis_utils workflows
to a running QGIS instance via the `qgis-mcp` plugin (by N. Karasiak).
Use cases: auto-reload regenerated layers, open output files in QGIS,
render preview maps, run Processing algorithms via QGIS.

## Setup (one-time, per machine)

### 1. QGIS Plugin

In QGIS: `Plugins → Manage and Install Plugins…` → search "**QGIS MCP**"
(by Nicolas Karasiak, repository nkarasiak/qgis-mcp) → install + enable.

Then: `Plugins → QGIS MCP → Start Server` (default port 9876).

Plugin version compatibility:
- Plugin v0.2.1 needs QGIS 3.28+ and Python 3.12+ (uses `datetime.UTC`)
- For older QGIS conda envs (Python 3.9): create a new env with
  `qgis>=3.44 python=3.12` — see "Common pitfalls" below

### 2. MCP Server (Python side)

Recommended: install via the upstream installer:

```bash
git clone https://github.com/nkarasiak/qgis-mcp.git ~/dev/Gunther-Schulz/qgis-mcp
cd ~/dev/Gunther-Schulz/qgis-mcp
uv sync
python install.py --non-interactive --clients claude-code
```

The installer prints a `claude mcp add` command — but on Linux, the
**suggested command needs PYTHONPATH** to find the qgis_mcp package
(the project isn't pip-packaged with src-layout).  Use this instead:

```bash
claude mcp add qgis --scope user \
  -e PYTHONPATH=/home/<USER>/dev/Gunther-Schulz/qgis-mcp/src \
  -- uv run --no-sync \
  --directory /home/<USER>/dev/Gunther-Schulz/qgis-mcp \
  src/qgis_mcp/server.py
```

(NOT in `~/.claude/settings.json` `mcpServers` block — Claude Code does
not read that field.  Use `claude mcp add` which writes to
`~/.claude.json`.)

### 3. gis_utils with optional `[qgis]` extra

```bash
pip install -e gis-utils[qgis]
# or, from a fresh PyPI install:  pip install gis-utils[qgis]
```

This pulls in `qgis-mcp` from git as the Python client dependency.
Without this extra, `gis_utils.qgis_bridge` becomes a silent no-op
(headless-safe).

### 4. Verify

```bash
claude mcp list                          # qgis: ✓ Connected
```

In Claude Code, after `/reload-plugins` or full restart, run
`mcp__qgis__diagnose` — should report all checks ✓.

## Usage patterns

### Pattern A — Auto-reload after `gis-workflow run`

Set the env var; the runner reloads each step's outputs in the open
QGIS project after the step succeeds.

```bash
GIS_WORKFLOW_QGIS_RELOAD=1 gis-workflow run
```

Layers in QGIS that point at the same source path get refreshed; new
files are NOT auto-added (they have to be in the project already).

### Pattern B — Programmatic from a project script

```python
from gis_utils import qgis_bridge

# Generate the file ...
gdf.to_file("Shape/MyResult.gpkg", driver="GPKG")

# Reload in QGIS (no-op if QGIS not running)
qgis_bridge.reload_paths(["Shape/MyResult.gpkg"])

# Or add as a new layer (idempotent — won't duplicate)
qgis_bridge.add_layer("Shape/MyResult.gpkg", name="My Result")

# Or arbitrary PyQGIS:
qgis_bridge.execute("iface.mapCanvas().refresh()")
```

### Pattern C — Claude orchestration via MCP tools directly

When working interactively in a Claude Code session with the qgis MCP
loaded, prefer `mcp__qgis__*` tools for ad-hoc layer manipulation,
canvas screenshots, layout exports, etc.  The bridge is for *workflow*
integration; the MCP tools are for *interactive* control.

## Common pitfalls (lessons learned the hard way)

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'UTC' from 'datetime'` when loading the plugin | QGIS conda env on Python 3.9; plugin needs 3.11+ for `datetime.UTC` | Create new env: `conda create -n qgis-ltr -c conda-forge qgis=3.44.7 python=3.12` and update Desktop launcher |
| `mcp__qgis__*` tools return `null` on every call | Wire-format mismatch — installed plugin is `nkarasiak/qgis-mcp` (length-prefixed framing) but registered MCP server is `jjsantos01/qgis_mcp` (newline-delimited) | Use the matching server: `nkarasiak/qgis-mcp` (51 tools, what's in the QGIS Plugin Registry today) |
| `ModuleNotFoundError: No module named 'qgis_mcp'` when starting the server | Project uses src-layout but is not pip-packaged | Set `PYTHONPATH=...src` via `claude mcp add -e PYTHONPATH=...` |
| `claude mcp list` doesn't show qgis even though `~/.claude/settings.json` has it | Claude Code does not read `mcpServers` from settings.json | Register via `claude mcp add` (writes to `~/.claude.json`) |
| Tools listed with old schemas after server change | Tool schemas cached per session | Full Claude Code restart — `/reload-plugins` is not enough |
| `is_available()` True but `reload_paths` returns 0 | No layers in the open QGIS project match the given paths | Either pre-load the layer (`qgis_bridge.add_layer(...)`) or expect this — the bridge does NOT auto-add files |

## Architecture quick-reference

```
gis_utils.qgis_bridge          ←  thin wrapper, lazy import of qgis-mcp client
        │
        │  via qgis-mcp's QgisMCPClient
        ▼
qgis-mcp Python server          ←  outside QGIS; speaks MCP and TCP
        │
        │  TCP socket :9876, length-prefixed JSON
        ▼
qgis-mcp QGIS plugin           ←  inside QGIS; QTimer-driven non-blocking server
        │
        │  PyQGIS API
        ▼
QGIS app
```

The bridge ships in gis_utils so workflow code can call it directly.
The qgis-mcp Python client is the optional dependency.  The QGIS plugin
must be installed separately via QGIS Plugin Manager.

## Discovery

```
mcp__gis-utils__catalog(search="qgis_bridge")
```

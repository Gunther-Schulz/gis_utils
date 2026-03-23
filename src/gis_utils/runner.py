"""
Simple YAML-based workflow runner for GIS projects.

Reads a workflow.yaml from a project directory, resolves step dependencies,
and executes scripts in the correct order. Steps can be marked as "always"
(run every time) or "once" (skip if all declared outputs already exist).

Usage:
    # From project directory:
    python -m gis_utils.runner

    # Or specify project path:
    python -m gis_utils.runner /path/to/project

    # Run a single step:
    python -m gis_utils.runner --step "Extract DXF layers"

    # Dry run (show execution plan):
    python -m gis_utils.runner --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml


def load_workflow(project_dir: Path) -> dict:
    """Load workflow.yaml from project directory."""
    wf_path = project_dir / "workflow.yaml"
    if not wf_path.exists():
        raise FileNotFoundError(f"No workflow.yaml found in {project_dir}")
    with open(wf_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_order(steps: list[dict]) -> list[dict]:
    """Topological sort of steps based on depends_on."""
    by_name = {s["name"]: s for s in steps}
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for s in steps:
        name = s["name"]
        if name not in in_degree:
            in_degree[name] = 0
        for dep in s.get("depends_on", []):
            if dep not in by_name:
                raise ValueError(f"Step '{name}' depends on unknown step '{dep}'")
            dependents[dep].append(name)
            in_degree[name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    ordered: list[str] = []

    while queue:
        # Stable sort: process in definition order among zero-degree nodes
        queue.sort(key=lambda n: next(i for i, s in enumerate(steps) if s["name"] == n))
        name = queue.pop(0)
        ordered.append(name)
        for dep in dependents[name]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(ordered) != len(steps):
        remaining = set(s["name"] for s in steps) - set(ordered)
        raise ValueError(f"Circular dependency involving: {remaining}")

    return [by_name[name] for name in ordered]


def should_skip(step: dict, project_dir: Path) -> bool:
    """Check if a 'once' step can be skipped (all outputs exist)."""
    if step.get("run", "always") != "once":
        return False
    outputs = step.get("outputs", [])
    if not outputs:
        return False
    return all((project_dir / out).exists() for out in outputs)


def run_step(step: dict, project_dir: Path, conda_env: str | None = None) -> bool:
    """Execute a single workflow step. Returns True on success."""
    script = step.get("script")
    if not script:
        print(f"  [skip] No script defined")
        return True

    script_path = project_dir / script
    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    args = step.get("args", [])
    if isinstance(args, str):
        args = args.split()

    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]

    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output",
               "python", str(script_path)] + [str(a) for a in args]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=False,
            timeout=step.get("timeout", 600),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timed out after {step.get('timeout', 600)}s")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def run_workflow(
    project_dir: str | Path,
    *,
    step_name: str | None = None,
    dry_run: bool = False,
    conda_env: str | None = None,
) -> bool:
    """
    Execute a project workflow.

    Args:
        project_dir: Path to project root (must contain workflow.yaml).
        step_name: Run only this step (and its dependencies). None = all.
        dry_run: If True, show plan without executing.
        conda_env: Conda environment name to run scripts in. None = current env.

    Returns:
        True if all steps succeeded.
    """
    project_dir = Path(project_dir).resolve()
    wf = load_workflow(project_dir)
    project_name = wf.get("project", {}).get("name", project_dir.name)
    default_env = wf.get("project", {}).get("conda_env")
    env = conda_env or default_env

    steps = wf.get("steps", [])
    if not steps:
        print("No steps defined in workflow.yaml")
        return True

    ordered = resolve_order(steps)

    # Filter to single step + dependencies if requested
    if step_name:
        needed = _collect_deps(step_name, {s["name"]: s for s in ordered})
        ordered = [s for s in ordered if s["name"] in needed]

    print(f"{'[DRY RUN] ' if dry_run else ''}Workflow: {project_name}")
    print(f"Steps: {len(ordered)}\n")

    all_ok = True
    for i, step in enumerate(ordered, 1):
        name = step["name"]
        mode = step.get("run", "always")
        skip = should_skip(step, project_dir)

        status = "SKIP (outputs exist)" if skip else mode
        prefix = f"[{i}/{len(ordered)}]"

        if dry_run:
            deps = step.get("depends_on", [])
            dep_str = f" (after: {', '.join(deps)})" if deps else ""
            print(f"  {prefix} {name} [{status}]{dep_str}")
            if step.get("script"):
                print(f"        → {step['script']}")
            continue

        if skip:
            print(f"  {prefix} {name} — skipped (outputs exist)")
            continue

        print(f"  {prefix} {name}...")
        t0 = time.time()
        ok = run_step(step, project_dir, conda_env=env)
        elapsed = time.time() - t0

        if ok:
            print(f"  {prefix} {name} — done ({elapsed:.1f}s)")
        else:
            print(f"  {prefix} {name} — FAILED ({elapsed:.1f}s)")
            if step.get("required", True):
                print(f"\nAborting: required step '{name}' failed.")
                return False
            all_ok = False

    if not dry_run:
        print(f"\n{'All steps completed.' if all_ok else 'Completed with errors.'}")
    return all_ok


def _collect_deps(name: str, by_name: dict[str, dict]) -> set[str]:
    """Collect a step and all its transitive dependencies."""
    if name not in by_name:
        raise ValueError(f"Unknown step: '{name}'")
    result = {name}
    for dep in by_name[name].get("depends_on", []):
        result |= _collect_deps(dep, by_name)
    return result


WORKFLOW_TEMPLATE = """\
project:
  name: {name}
  conda_env: gis

steps:
  - name: Example step
    script: scripts/example.py
    run: always
    # outputs:
    #   - output/result.shp
    # depends_on:
    #   - Other step name
"""

CLAUDE_MD_CONTENT = """\
# {name}

## Project notes
<!-- Add project-specific context here: CRS, coordinate quirks, data sources, etc. -->

## Environment
- Conda env: `gis`
- Workflow: `gis-workflow` (run), `gis-workflow --dry-run` (plan), `gis-workflow --step "Name"` (single step)
- Scripts in `scripts/`, old scripts in `deprecated_scripts/` (frozen, do not modify)

## gis_utils library
This project uses `gis_utils` (pip install -e ~/dev/Gunther-Schulz/gis_utils).
ALWAYS use it instead of writing equivalent code from scratch.

### Rules
- **Markdown tables:** ALWAYS use `from gis_utils import markdown_table` — fixed-width columns that align in raw markdown. NEVER use tabulate, pandas .to_markdown(), or manual formatting.
- **DXF extraction:** use `extract_dxf_layers()` / `extract_dxf_circles()` — handles bulge/arc interpolation, block recursion, all entity types. NEVER iterate DXF entities manually.
- **DXF creation:** use `new_dxf_document()` and `ensure_layer()` for CAD-compatible headers.
- **Geometry repair:** use `make_valid_gdf()`, not inline shapely calls.
- **Removing holes:** use `remove_inner_rings()`.
- **Set difference:** use `subtract_geometries()`.
- **Overlap removal:** use `subtract_smaller_overlaps()`.
- **Area reports:** use `area_report()`, `intersection_areas()`, `area_by_category()`.
- **Load + union:** use `load_and_union()` to avoid double-counting overlapping polygons.
- **Find columns:** use `find_column(gdf, candidates)` for varying column name conventions.

### Full API
See ~/dev/Gunther-Schulz/gis_utils/CLAUDE.md
"""


def init_project(project_dir: str | Path) -> None:
    """
    Initialize a GIS project with workflow.yaml, scripts/, and CLAUDE.md.

    Args:
        project_dir: Path to project directory.
    """
    project_dir = Path(project_dir).resolve()
    name = project_dir.name

    # Create scripts/
    scripts_dir = project_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    print(f"  scripts/ — {'exists' if scripts_dir.exists() else 'created'}")

    # Create workflow.yaml (don't overwrite)
    wf_path = project_dir / "workflow.yaml"
    if wf_path.exists():
        print(f"  workflow.yaml — already exists (skipped)")
    else:
        wf_path.write_text(WORKFLOW_TEMPLATE.format(name=name), encoding="utf-8")
        print(f"  workflow.yaml — created")

    # Create CLAUDE.md (don't overwrite)
    claude_path = project_dir / "CLAUDE.md"
    if claude_path.exists():
        print(f"  CLAUDE.md — already exists (skipped)")
    else:
        claude_path.write_text(CLAUDE_MD_CONTENT.format(name=name), encoding="utf-8")
        print(f"  CLAUDE.md — created")

    print(f"\nProject initialized: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="GIS project workflow runner.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # `gis-workflow run` (default when no subcommand)
    run_parser = subparsers.add_parser("run", help="Run workflow (default)")
    run_parser.add_argument(
        "project_dir", nargs="?", default=".",
        help="Project directory (default: current directory)",
    )
    run_parser.add_argument("--step", "-s", default=None, help="Run only this step (and its dependencies)")
    run_parser.add_argument("--dry-run", "-n", action="store_true", help="Show execution plan without running")
    run_parser.add_argument("--env", "-e", default=None, help="Conda environment (overrides workflow.yaml)")

    # `gis-workflow init`
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument(
        "project_dir", nargs="?", default=".",
        help="Project directory (default: current directory)",
    )

    args = parser.parse_args()

    # Default to "run" if no subcommand given but args look like a path
    if args.command is None:
        # Re-parse as "run" with remaining args
        args = run_parser.parse_args()
        args.command = "run"

    if args.command == "init":
        init_project(args.project_dir)
        return

    ok = run_workflow(
        args.project_dir,
        step_name=args.step,
        dry_run=args.dry_run,
        conda_env=args.env,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

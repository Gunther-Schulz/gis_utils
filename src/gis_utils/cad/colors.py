"""
ACI (AutoCAD Color Index) resolution for the CAD emitter.

Colour handling is salvaged (referenced, not imported) from
``Python-ACAD-Tools/src/dxf_utils.py`` (``get_color_code`` /
``convert_transparency``) and its ``aci_colors.yaml`` table, decoupled from
that project's ``ProjectLoader``: the resolver takes a plain name→ACI mapping
built from the packaged colour table instead of a project object.

A colour value in a style may be:

* an ``int`` — used verbatim as an ACI code (1..255),
* a colour ``name`` (case-insensitive, e.g. ``"red"``, ``"vermilion-light"``),
  resolved to its ACI code via the packaged table, or
* an ``"R,G,B"`` string or ``[r, g, b]`` sequence — a true-colour RGB tuple.

Unlike the salvaged original, an **unknown colour name is a hard error**
(:class:`ColorError`), not a silent fallback to white — see the gis-safety
"no silent fallbacks" rule.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import yaml

# ACI code returned for BYLAYER-style "no explicit colour". 7 is the CAD
# convention for the default fore/background-adaptive colour.
DEFAULT_ACI = 7

# A resolved colour is either an ACI index or a true-colour RGB triple.
Color = int  # or tuple[int, int, int]; kept as a doc alias for readability


class ColorError(ValueError):
    """Raised for an unresolvable colour value (unknown name, bad RGB)."""


@lru_cache(maxsize=1)
def load_aci_table() -> list[dict]:
    """
    Load the packaged ACI colour table (``aci_colors.yaml``).

    Returns a list of ``{"aciCode": int, "name": str, "rgb": [r, g, b]}`` dicts.
    Table salvaged verbatim from Python-ACAD-Tools.
    """
    text = files("gis_utils.cad").joinpath("aci_colors.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@lru_cache(maxsize=1)
def name_to_aci() -> dict[str, int]:
    """Case-insensitive colour-name → ACI-code mapping from the packaged table."""
    mapping: dict[str, int] = {}
    for entry in load_aci_table():
        name = entry.get("name")
        code = entry.get("aciCode")
        if name is not None and code is not None:
            mapping[str(name).lower()] = int(code)
    return mapping


@lru_cache(maxsize=1)
def aci_to_rgb() -> dict[int, tuple[int, int, int]]:
    """ACI-code → (r, g, b) mapping from the packaged table."""
    mapping: dict[int, tuple[int, int, int]] = {}
    for entry in load_aci_table():
        code = entry.get("aciCode")
        rgb = entry.get("rgb")
        if code is not None and rgb and len(rgb) == 3:
            mapping[int(code)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return mapping


def resolve_color(value) -> int | tuple[int, int, int]:
    """
    Resolve a style colour value to an ACI code or an RGB tuple.

    Salvaged from ``dxf_utils.get_color_code`` but made strict: an unknown
    colour *name* raises :class:`ColorError` instead of defaulting to white.

    Args:
        value: ``None`` → :data:`DEFAULT_ACI`; ``int`` → ACI as-is; ``"R,G,B"``
            string → RGB tuple; colour name → ACI via the packaged table;
            3-sequence → RGB tuple.

    Returns:
        An ``int`` ACI code, or a ``(r, g, b)`` tuple for true colour.
    """
    if value is None:
        return DEFAULT_ACI
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ColorError(f"Invalid colour value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if "," in value:
            try:
                parts = tuple(int(p) for p in value.split(","))
            except ValueError as exc:
                raise ColorError(f"Invalid RGB colour string: {value!r}") from exc
            if len(parts) != 3:
                raise ColorError(f"RGB colour string needs 3 components: {value!r}")
            return parts
        code = name_to_aci().get(value.lower())
        if code is None:
            raise ColorError(
                f"Unknown colour name {value!r}. Not in the ACI colour table."
            )
        return code
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(c) for c in value)
    raise ColorError(f"Unsupported colour value: {value!r}")


def normalize_transparency(value) -> float | None:
    """
    Clamp a transparency value to ``0.0..1.0`` (0 = opaque, 1 = fully clear).

    Salvaged from ``dxf_utils.convert_transparency``. Returns ``None`` for an
    unparseable value so the caller can leave transparency unset.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 1.0)
    if isinstance(value, str):
        try:
            return min(max(float(value), 0.0), 1.0)
        except ValueError:
            return None
    return None

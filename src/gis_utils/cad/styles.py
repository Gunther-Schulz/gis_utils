"""
Strict style schema and loader for the CAD emitter (``cad_styles.yaml``).

This is a **new, strict** format — a normalised, closed-field-set successor to
Python-ACAD-Tools' ``styles.yaml`` (structure erhoben and referenced, not
imported). Differences from the reference, all deliberate:

* **snake_case, closed field menus.** camelCase reference keys are renamed
  (``linetypeScale`` → ``linetype_scale``, ``individual_hatches`` →
  ``individual``, ``attachmentPoint`` → ``attachment``, ``maxWidth`` →
  ``max_width``). Any unknown field is a hard :class:`StyleError`, so a typo
  fails loudly instead of being silently ignored (the reference only warned).
* **``schema_version`` required.** The top-level document must declare
  ``schema_version: 1``; a mismatch is refused rather than best-guessed.
* **No preset/override/inline merging.** A layer references a style by name;
  presets-with-overrides and inline styles from the reference are dropped.

File shape::

    schema_version: 1
    styles:
      <style-name>:
        layer:  {color, linetype, lineweight, plot, locked, frozen, is_on,
                 transparency, linetype_scale}
        hatch:  {pattern, scale, angle, color, transparency, individual}
        text:   {color, height, font, max_width, attachment, rotation}
        entity: {close, linetype_scale, linetype_generation}

Every block and every field is optional; only what is present is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA_VERSION = 1

# Closed field menus. Unknown keys at any level are a hard error.
_LAYER_FIELDS = frozenset(
    {"color", "linetype", "lineweight", "plot", "locked", "frozen", "is_on",
     "transparency", "linetype_scale"}
)
_HATCH_FIELDS = frozenset(
    {"pattern", "scale", "angle", "color", "transparency", "individual"}
)
_TEXT_FIELDS = frozenset(
    {"color", "height", "font", "max_width", "attachment", "rotation"}
)
_ENTITY_FIELDS = frozenset(
    {"close", "linetype_scale", "linetype_generation"}
)
_STYLE_FIELDS = frozenset({"layer", "hatch", "text", "entity"})

# MTEXT attachment-point names → ezdxf integer codes (salvaged from
# dxf_utils._apply_text_style_properties). The closed set of accepted values.
ATTACHMENT_CODES = {
    "TOP_LEFT": 1, "TOP_CENTER": 2, "TOP_RIGHT": 3,
    "MIDDLE_LEFT": 4, "MIDDLE_CENTER": 5, "MIDDLE_RIGHT": 6,
    "BOTTOM_LEFT": 7, "BOTTOM_CENTER": 8, "BOTTOM_RIGHT": 9,
}


class StyleError(ValueError):
    """Raised for a malformed or unrecognised style definition."""


@dataclass(frozen=True)
class LayerStyle:
    color: object | None = None
    linetype: str | None = None
    lineweight: int | None = None
    plot: bool | None = None
    locked: bool | None = None
    frozen: bool | None = None
    is_on: bool | None = None
    transparency: float | None = None
    linetype_scale: float | None = None


@dataclass(frozen=True)
class HatchStyle:
    pattern: str = "SOLID"
    scale: float = 1.0
    angle: float = 0.0
    color: object | None = None
    transparency: float | None = None
    individual: bool = False


@dataclass(frozen=True)
class TextStyle:
    color: object | None = None
    height: float = 2.5
    font: str = "Standard"
    max_width: float | None = None
    attachment: str | None = None
    rotation: float | None = None


@dataclass(frozen=True)
class EntityStyle:
    close: bool | None = None
    linetype_scale: float | None = None
    linetype_generation: bool | None = None


@dataclass(frozen=True)
class Style:
    name: str
    layer: LayerStyle | None = None
    hatch: HatchStyle | None = None
    text: TextStyle | None = None
    entity: EntityStyle | None = None


def _reject_unknown(block: dict, allowed: frozenset, where: str) -> None:
    """Raise :class:`StyleError` if *block* carries any key outside *allowed*."""
    if not isinstance(block, dict):
        raise StyleError(f"{where}: expected a mapping, got {type(block).__name__}")
    unknown = set(block) - allowed
    if unknown:
        raise StyleError(
            f"{where}: unknown field(s) {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )


def _build_text(block: dict, where: str) -> TextStyle:
    _reject_unknown(block, _TEXT_FIELDS, where)
    attachment = block.get("attachment")
    if attachment is not None and str(attachment).upper() not in ATTACHMENT_CODES:
        raise StyleError(
            f"{where}: unknown attachment {attachment!r}; "
            f"allowed: {sorted(ATTACHMENT_CODES)}"
        )
    return TextStyle(
        color=block.get("color"),
        height=block.get("height", 2.5),
        font=block.get("font", "Standard"),
        max_width=block.get("max_width"),
        attachment=str(attachment).upper() if attachment is not None else None,
        rotation=block.get("rotation"),
    )


def _build_style(name: str, sdef: dict) -> Style:
    _reject_unknown(sdef, _STYLE_FIELDS, f"style '{name}'")

    layer = None
    if "layer" in sdef:
        _reject_unknown(sdef["layer"], _LAYER_FIELDS, f"style '{name}'.layer")
        layer = LayerStyle(**sdef["layer"])

    hatch = None
    if "hatch" in sdef:
        _reject_unknown(sdef["hatch"], _HATCH_FIELDS, f"style '{name}'.hatch")
        hatch = HatchStyle(**sdef["hatch"])

    text = None
    if "text" in sdef:
        text = _build_text(sdef["text"], f"style '{name}'.text")

    entity = None
    if "entity" in sdef:
        _reject_unknown(sdef["entity"], _ENTITY_FIELDS, f"style '{name}'.entity")
        entity = EntityStyle(**sdef["entity"])

    return Style(name=name, layer=layer, hatch=hatch, text=text, entity=entity)


def parse_styles(data: dict) -> dict[str, Style]:
    """
    Validate and build the style map from a parsed ``cad_styles.yaml`` document.

    Args:
        data: The mapping loaded from YAML (must carry ``schema_version`` and
            a ``styles`` mapping).

    Returns:
        ``{style_name: Style}``.

    Raises:
        StyleError: on a missing/wrong ``schema_version``, a missing ``styles``
            block, or any unknown field.
    """
    if not isinstance(data, dict):
        raise StyleError("cad_styles: top level must be a mapping")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise StyleError(
            f"cad_styles: schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )
    styles_block = data.get("styles")
    if not isinstance(styles_block, dict):
        raise StyleError("cad_styles: missing or malformed 'styles' mapping")

    return {name: _build_style(name, sdef) for name, sdef in styles_block.items()}


def load_styles(path: str | Path) -> dict[str, Style]:
    """
    Load and validate a ``cad_styles.yaml`` file into a ``{name: Style}`` map.

    Args:
        path: Path to the style file.

    Raises:
        FileNotFoundError: if *path* does not exist.
        StyleError: on any schema violation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Style file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_styles(data)

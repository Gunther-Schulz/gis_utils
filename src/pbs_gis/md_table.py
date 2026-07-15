"""
Reusable markdown table formatting with fixed-width columns
so tables align in raw markdown (source view) and render correctly.
"""

from typing import Any, List, Optional, Sequence, Union


def _format_cell(cell: Any, number_format: str) -> str:
    """Convert cell to string; format numbers with number_format."""
    if cell is None:
        return ""
    if isinstance(cell, (int, float)):
        try:
            return format(cell, number_format)
        except (ValueError, TypeError):
            return str(cell)
    return str(cell)


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    align: Optional[Sequence[str]] = None,
    number_format: str = ",.0f",
    max_col_width: Optional[Union[int, Sequence[Optional[int]]]] = None,
) -> str:
    """
    Build a markdown table string with fixed-width columns so it aligns in raw markdown.

    Args:
        headers: Column header strings.
        rows: List of rows; each row is a sequence of cell values (str, int, float, or None).
        align: Per-column alignment: 'l' (left), 'c' (center), 'r' (right).
               Default: first column 'l', rest 'r'.
        number_format: Format spec for numeric cells (e.g. ",.0f" or ",.2f").
        max_col_width: Cap column width(s). Int for all columns, or list of int/None per column.

    Returns:
        Full table string (header + separator + data rows), including trailing newline.
    """
    ncols = len(headers)
    if ncols == 0:
        return ""

    # Default alignment: first left, rest right
    if align is None:
        align = ["l"] + ["r"] * (ncols - 1)
    align = list(align)
    while len(align) < ncols:
        align.append("r")

    # Format all cells and compute column widths
    formatted: List[List[str]] = []
    for row in rows:
        if len(row) != ncols:
            row = list(row) + [""] * (ncols - len(row))
        formatted.append([_format_cell(c, number_format) for c in row])

    widths = [len(h) for h in headers]
    for row in formatted:
        for j, s in enumerate(row):
            if j < len(widths):
                widths[j] = max(widths[j], len(s))

    # Apply max_col_width
    if max_col_width is not None:
        if isinstance(max_col_width, int):
            widths = [min(w, max_col_width) for w in widths]
        else:
            for j, cap in enumerate(max_col_width):
                if j < len(widths) and cap is not None:
                    widths[j] = min(widths[j], cap)

    def pad(s: str, w: int, a: str) -> str:
        if len(s) > w:
            return s[: w - 1] + "…"
        if a == "l":
            return s.ljust(w)
        if a == "r":
            return s.rjust(w)
        return s.center(w)

    # Separator line: each cell is " " + content + " " (width w+2); match that
    sep_parts = []
    for j, (w, a) in enumerate(zip(widths, align)):
        n = w + 2  # total cell width in our format
        if a == "l":
            sep_cell = ":" + "-" * (n - 1)
        elif a == "c":
            sep_cell = ":" + "-" * max(0, n - 2) + ":"
        else:
            sep_cell = "-" * (n - 1) + ":"
        sep_parts.append(sep_cell)
    sep_line = "|" + "|".join(sep_parts) + "|\n"

    out = []
    # Header
    header_cells = [pad(headers[j], widths[j], align[j]) for j in range(ncols)]
    out.append("| " + " | ".join(header_cells) + " |\n")
    out.append(sep_line)
    # Data rows
    for row in formatted:
        cells = [pad(row[j], widths[j], align[j]) for j in range(ncols)]
        out.append("| " + " | ".join(cells) + " |\n")

    return "".join(out)

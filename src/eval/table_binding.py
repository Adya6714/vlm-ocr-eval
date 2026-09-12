"""
Header–cell binding accuracy for tables (Decision #12).

Why this metric and not table-to-prose: generating a sentence per row
needs its own evaluation methodology this project does not have budget
for. Binding is checkable: after OCR, is each data cell still attached
to the correct column header? The renderer (once table-embedded layouts
exist) can supply clean ground truth. Sarvam Extract's per-field
records are the production analogue (Decision #13).

Why IoU / span overlap rather than string match alone: OCR may mangle
the header *text* while still putting the cell in the right column.
The primary score is geometric (predicted cell box vs GT header box).
A secondary exact-header-string score is reported separately so a
caller can see recognition vs structure.

The live layout bank is PARTIAL: almost no `table-embedded` pages, no
`form`, and `render.py` does not paint table cells into GT. This module
therefore scores *records you pass in*, including synthetic fixtures
in tests. It does not pretend the bank already has table GT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HeaderBox:
    """One column header. `col` is 0-based left-to-right."""

    col: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class CellBox:
    """One data cell. Binding target is the header in the same column."""

    row: int
    col: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def _x_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def bind_cell_to_header(cell: CellBox, headers: Sequence[HeaderBox]) -> int:
    """
    Assign a cell to the header with maximum horizontal overlap.

    Ties break to the smaller column index (stable, not learned). This
    is the geometry baseline Chapter 5 describes; it will look easy on
    unrotated synthetic grids and is expected to fail on skewed scans.
    """
    if not headers:
        raise ValueError("no headers to bind to")
    best_col = headers[0].col
    best_ov = -1.0
    cell_span = (cell.x0, cell.x1)
    for h in headers:
        ov = _x_overlap(cell_span, (h.x0, h.x1))
        if ov > best_ov or (ov == best_ov and h.col < best_col):
            best_ov = ov
            best_col = h.col
    return best_col


def binding_accuracy(
    cells: Sequence[CellBox],
    headers: Sequence[HeaderBox],
    pred_cols: Sequence[int] | None = None,
) -> dict:
    """
    Fraction of data cells bound to the correct GT column.

    If pred_cols is None, use bind_cell_to_header (geometry oracle on
    the given boxes). If pred_cols is provided, it is the model's
    predicted column index per cell, in the same order as `cells`.

    Also reports exact header-string match when pred_cols is used with
    headers that still have readable text — optional, not the headline.
    """
    if not cells:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "string_match": None,
        }
    if pred_cols is None:
        pred_cols = [bind_cell_to_header(c, headers) for c in cells]
    if len(pred_cols) != len(cells):
        raise ValueError("pred_cols length must match cells")
    header_by_col = {h.col: h for h in headers}
    correct = 0
    string_ok = 0
    string_n = 0
    for cell, pred in zip(cells, pred_cols):
        if pred == cell.col:
            correct += 1
        gt_h = header_by_col.get(cell.col)
        pred_h = header_by_col.get(pred)
        if gt_h is not None and pred_h is not None:
            string_n += 1
            if pred_h.text == gt_h.text and pred == cell.col:
                string_ok += 1
    return {
        "n": len(cells),
        "correct": correct,
        "accuracy": correct / len(cells),
        "string_match": (string_ok / string_n) if string_n else None,
    }

"""
Pairwise reading-order module (count-and-sort, Chapter 5).

Why pairwise, not a pointer net: Kendall tau is already a pairwise
agreement count. Training (when we have labels) and evaluation share
that shape. A pointer network is the documented later alternative
(IMPLEMENTATION.md Stage 2b) and is not built here.

Why geometric features only: the layout bank does not yet have enough
two-column / table-embedded pages to train a serious classifier
(PARTIAL: 25 single-column, 2 two-column, 1 marginalia, 0 form). A
learned model on that bank would overfit single-column top-to-bottom.

Inference: for every pair (i,j), vote "i before j" if i is above j, or
same band and i is to the left. Sort by how often a block is predicted
to precede others.

Scored by: eval/reading_order_metric.kendall_tau_order.
"""
from __future__ import annotations

from typing import Sequence


def pairwise_before(a: dict, b: dict, y_tol: float = 0.02) -> bool:
    """
    True if block a should be read before block b under a simple
    top-to-bottom then left-to-right rule.

    y_tol (page fraction) treats nearly-aligned blocks as the same band
    so two-column cells on one baseline sort by x, not by 1px jitter.
    """
    if a["y"] + y_tol < b["y"]:
        return True
    if b["y"] + y_tol < a["y"]:
        return False
    return a["x"] < b["x"]


def count_sort_order(blocks: Sequence[dict], y_tol: float = 0.02) -> list[str]:
    """Return block ids in predicted reading order."""
    ids = [str(b["id"]) for b in blocks]
    by_id = {str(b["id"]): b for b in blocks}
    scores = {i: 0 for i in ids}
    for i, id_i in enumerate(ids):
        for id_j in ids[i + 1 :]:
            if pairwise_before(by_id[id_i], by_id[id_j], y_tol=y_tol):
                scores[id_i] += 1
            else:
                scores[id_j] += 1
    return sorted(ids, key=lambda i: (-scores[i], i))

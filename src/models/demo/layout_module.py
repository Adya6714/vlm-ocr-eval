"""
Block-level layout helper for the demo.

Why it is not a trained detector yet: Stage 1's bank is PARTIAL (no
form, no populated table-embedded category for sampling). Training a
detector on 25 single-column templates would not test the Stage 3
curve. This file converts renderer PageGT *lines* into the block dicts
reading_order_module and Kendall tau consume — so the metric stack can
run on real renderer output today, with an oracle (GT boxes) not a
learned detector.

When the bank has form / table-embedded pages, replace
`blocks_from_page_gt` with a detector trained on those boxes; keep the
same dict schema (`id`, `x`, `y`, `w`, `h`, `region`).
"""
from __future__ import annotations

from typing import Any


def blocks_from_page_gt(gt: dict[str, Any]) -> list[dict]:
    """
    One block per LineGT (or cluster region if lines are missing).

    Coordinates stay in pixel space; only relative order matters for
    the pairwise module. `id` is stable: region + line index.
    """
    width = max(int(gt.get("width") or 1), 1)
    height = max(int(gt.get("height") or 1), 1)
    lines = gt.get("lines") or []
    blocks = []
    if lines:
        for line in lines:
            x0, y0, x1, y1 = line["bbox"]
            blocks.append({
                "id": f"{line.get('region', 'r')}_{line.get('reading_order', 0)}",
                "x": x0 / width,
                "y": y0 / height,
                "w": max(x1 - x0, 1) / width,
                "h": max(y1 - y0, 1) / height,
                "region": line.get("region"),
                "text": line.get("text"),
                "gt_reading_order": line.get("reading_order"),
            })
        return blocks
    # Fallback: unique regions from clusters
    seen = {}
    for cl in gt.get("clusters") or []:
        region = cl.get("region", "body")
        if region in seen:
            continue
        x0, y0, x1, y1 = cl["bbox"]
        seen[region] = {
            "id": region,
            "x": x0 / width,
            "y": y0 / height,
            "w": max(x1 - x0, 1) / width,
            "h": max(y1 - y0, 1) / height,
            "region": region,
            "gt_reading_order": cl.get("reading_order"),
        }
    return [seen[k] for k in sorted(seen)]

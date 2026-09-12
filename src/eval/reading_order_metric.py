"""
Kendall tau between predicted and ground-truth block reading order.

Why tau and not accuracy: reading order is a permutation problem, not
a classification problem. Accuracy would treat "swapped two adjacent
blocks" the same as "read the page in reverse," which is not useful
signal. Tau counts pairwise order agreements, so the score is graded.

Why per layout-complexity bucket: Stage 3's acceptance criterion is a
tau-vs-complexity *curve* (single-column → two-column → marginalia →
table-embedded), not one number. A single mean would hide a system that
only fails when the page gets hard. Bucket labels match
`renderer.layout_sources.CATEGORIES`.

Called from: Stage 3 eval of any model's ordered block ids (demo,
instrument, or a geometric baseline). Does not require the demo LoRA
stack to exist. Feeds `tau_by_bucket` for the curve.

Does not invent table/form pages: if a bucket has no documents, that
bucket is omitted from the curve and listed as empty — the layout bank
is PARTIAL (no `form`, almost no `table-embedded`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from scipy.stats import kendalltau

# Same axis as layout_sources.CATEGORIES, in increasing complexity.
COMPLEXITY_ORDER = (
    "single-column",
    "two-column",
    "marginalia",
    "table-embedded",
    "form",
)


def permutation_ranks(
    gt_order: Sequence[str],
    pred_order: Sequence[str],
) -> list[int]:
    """
    Map a predicted id sequence onto ground-truth ranks.

    gt_order is the true reading sequence of block ids. pred_order must
    be a permutation of the same ids (duplicates and missing ids are
    errors — silently dropping a block would let a model game tau by
    omitting hard regions; coverage belongs in RLVR, not here).
    """
    if len(gt_order) != len(pred_order):
        raise ValueError(
            f"pred has {len(pred_order)} blocks, gt has {len(gt_order)}; "
            "reading-order tau requires the same set, not a subset"
        )
    if set(gt_order) != set(pred_order):
        extra = set(pred_order) - set(gt_order)
        missing = set(gt_order) - set(pred_order)
        raise ValueError(f"id mismatch extra={extra!r} missing={missing!r}")
    pos = {block_id: i for i, block_id in enumerate(gt_order)}
    return [pos[block_id] for block_id in pred_order]


def kendall_tau_order(
    gt_order: Sequence[str],
    pred_order: Sequence[str],
) -> float:
    """
    Kendall tau-b for one page.

    Returns 1.0 for the identity permutation, -1.0 for a full reversal
    (n≥2). One block: tau is undefined in scipy (nan); we return 1.0
    because a singleton has no pairwise mistakes. Empty page: 1.0 for
    the same reason (nothing to disorder).
    """
    n = len(gt_order)
    if n <= 1:
        if list(gt_order) != list(pred_order):
            raise ValueError("singleton/empty id mismatch")
        return 1.0
    ranks = permutation_ranks(gt_order, pred_order)
    tau, _ = kendalltau(list(range(n)), ranks)
    if tau != tau:  # NaN from all-ties, should not happen without ties
        raise ValueError("kendalltau returned NaN")
    return float(tau)


def tau_by_bucket(
    pages: Iterable[Mapping],
) -> dict[str, dict]:
    """
    Mean tau per layout-complexity bucket.

    Each page mapping needs:
      - gt_order: list[str]
      - pred_order: list[str]
      - category: one of COMPLEXITY_ORDER

    Empty buckets are present with n=0 and tau=None so a caller can
    plot a hole instead of silently averaging them away.
    """
    buckets: dict[str, list[float]] = {c: [] for c in COMPLEXITY_ORDER}
    unknown: list[str] = []
    for page in pages:
        cat = page["category"]
        tau = kendall_tau_order(page["gt_order"], page["pred_order"])
        if cat not in buckets:
            unknown.append(cat)
            buckets.setdefault(cat, []).append(tau)
        else:
            buckets[cat].append(tau)
    out: dict[str, dict] = {}
    for cat, values in buckets.items():
        out[cat] = {
            "n": len(values),
            "mean_tau": (sum(values) / len(values)) if values else None,
            "taus": values,
        }
    if unknown:
        out["_unknown_categories"] = sorted(set(unknown))
    return out


def geometric_pred_order(
    blocks: Sequence[Mapping],
    *,
    rtl: bool = False,
) -> list[str]:
    """
    Top-to-bottom, then left-to-right (or right-to-left) sort.

    This is the naive baseline Chapter 5 says does *not* solve Indic
    multi-column pages — it exists so tau-vs-complexity can show when
    it fails. Each block needs id, y, x (page-fraction or pixels;
    only relative order matters).
    """
    sign = -1 if rtl else 1
    ordered = sorted(blocks, key=lambda b: (float(b["y"]), sign * float(b["x"])))
    return [str(b["id"]) for b in ordered]


def inventory_layout_bank(bank_path) -> dict:
    """
    What the live bank can actually support for Stage 3.

    Region-level reading_order exists on every template. Cell-level
    table GT does not: tiny Wikipedia infobox `kind=table` boxes are
    not column grids, and render.py never paints table/form regions.
    """
    import json
    from pathlib import Path

    path = Path(bank_path)
    templates = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {c: 0 for c in COMPLEXITY_ORDER}
    n_table_kind = 0
    n_form_field = 0
    n_real_table = 0
    pages = []
    for tmpl in templates:
        cat = tmpl.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
        regions = tmpl.get("regions") or []
        for r in regions:
            if r.get("kind") == "table":
                n_table_kind += 1
                if r.get("height", 0) >= 0.18 and r.get("width", 0) >= 0.25:
                    n_real_table += 1
            if r.get("kind") == "form_field":
                n_form_field += 1
        gt_order = [
            r["name"]
            for r in sorted(regions, key=lambda x: int(x.get("reading_order", 0)))
        ]
        blocks = [
            {"id": r["name"], "x": float(r["x"]), "y": float(r["y"])}
            for r in regions
        ]
        if not blocks:
            continue
        pred = geometric_pred_order(blocks)
        pages.append({
            "category": cat,
            "gt_order": gt_order,
            "pred_order": pred,
            "source_id": tmpl.get("source_id"),
            "page_index": tmpl.get("page_index"),
        })
    curve = tau_by_bucket(pages)
    compact_curve = {
        cat: {"n": rec["n"], "mean_tau": rec["mean_tau"]}
        for cat, rec in curve.items()
        if cat != "_unknown_categories"
    }
    if "_unknown_categories" in curve:
        compact_curve["_unknown_categories"] = curve["_unknown_categories"]
    return {
        "n_templates": len(templates),
        "n_pages_scored": len(pages),
        "category_counts": counts,
        "n_table_kind_regions": n_table_kind,
        "n_real_table_regions": n_real_table,
        "n_form_field_regions": n_form_field,
        "geometric_baseline_tau": compact_curve,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    bank = Path(__file__).resolve().parents[2] / "data" / "cache" / "layouts" / "bank.json"
    report = inventory_layout_bank(bank)
    print(json.dumps(report, indent=2, default=str))

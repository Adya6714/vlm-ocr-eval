"""
src/probes/cascade.py

Stage 6 — triage cascade (Decision #16), offline on Stage 5 cache.

Why this file exists
--------------------
Probe 5b's mean_confidence is a candidate *router*: low confidence →
escalate to a perfect reviewer (human / production OCR treated as
CER=0 on those pages); high confidence → keep the page and eat its
production CER. Decision #16 asks whether that router is better than
random, layout-complexity, and Tesseract-confidence at the **same**
escalation count. Cost savings are not the headline.

No new Sarvam calls (Decision #19). Requires Stage 5b's paired table
to exist — by default the Hindi plains that have Sarvam text in
`sarvam_transfer_probe.jsonl` and/or `sarvam_stage5b_pages.jsonl`.

Called from: CLI; tests/test_cascade.py with synthetic CERs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SRC / "eval") not in sys.path:
    sys.path.insert(0, str(_SRC / "eval"))
if str(_SRC / "analysis") not in sys.path:
    sys.path.insert(0, str(_SRC / "analysis"))

from analysis.grapheme_cer_rescore import grapheme_clusters  # noqa: E402
from eval.equivalence_tables import normalize_tier1  # noqa: E402
from eval.transfer_analysis import (  # noqa: E402
    instrument_confidence_by_id,
    load_engine_rows,
    load_sarvam_rows,
    pair_sarvam_cohort,
)

RANDOM_SEED = 0
N_RANDOM_DRAWS = 1000


def residual_system_cer(cers: list[float], escalate: np.ndarray) -> float:
    """
    Mean CER if escalated pages are reviewed perfectly (error 0).

    Why this metric: Stage 6 compares routers, not OCR engines. A fair
    comparison holds the escalation *count* fixed; the better router
    sends the actually-wrong pages to the reviewer, so leftover error
    on the kept set (and therefore on the whole set with zeros filled
    in) is lower.
    """
    cers_a = np.asarray(cers, dtype=float)
    keep = ~np.asarray(escalate, dtype=bool)
    if len(cers_a) == 0:
        return float("nan")
    filled = np.where(keep, cers_a, 0.0)
    return float(filled.mean())


def escalate_lowest_scores(scores: list[float], k: int) -> np.ndarray:
    """Escalate the k smallest scores (ties: stable argsort)."""
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    mask[order[:k]] = True
    return mask


def escalate_highest_scores(scores: list[float], k: int) -> np.ndarray:
    """Escalate the k largest scores (layout complexity / line length)."""
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    mask[order[:k]] = True
    return mask


def random_mean_residual(
    cers: list[float], k: int, n_draws: int = N_RANDOM_DRAWS, seed: int = RANDOM_SEED
) -> float:
    """Expected residual CER if a random k pages are escalated."""
    n = len(cers)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    acc = []
    cers_a = np.asarray(cers, dtype=float)
    for _ in range(n_draws):
        mask = np.zeros(n, dtype=bool)
        if k > 0:
            idx = rng.choice(n, size=min(k, n), replace=False)
            mask[idx] = True
        acc.append(residual_system_cer(cers_a.tolist(), mask))
    return float(np.mean(acc))


def build_cascade_rows(repo_root: Path) -> list[dict]:
    """
    One row per Hindi plain page that has Sarvam CER + instrument confidence.

    Layout score = GT grapheme count (these items are single-line crops,
    not Stage 3 page-layout buckets — Decision #89).
    """
    sarvam_rows = load_sarvam_rows(
        [
            repo_root / "data/probe_results/sarvam_transfer_probe.jsonl",
            repo_root / "data/probe_results/sarvam_stage5b_pages.jsonl",
        ]
    )
    paired = pair_sarvam_cohort(
        repo_root, sarvam_rows, script="hindi", variant="plain"
    )
    conf = instrument_confidence_by_id(repo_root, condition="hindi")
    tess = load_engine_rows(repo_root, "tesseract", "hindi")
    rows = []
    for rec in paired:
        iid = rec["image_id"]
        if iid not in conf:
            continue
        gt = rec.get("ground_truth") or ""
        tess_rec = tess.get((iid, "plain"))
        tess_conf = None if tess_rec is None else tess_rec.get("confidence")
        rows.append(
            {
                "image_id": iid,
                "sarvam_cer": rec["sarvam_cer"],
                "instrument_confidence": conf[iid],
                "layout_n_graphemes": len(grapheme_clusters(normalize_tier1(gt))),
                "tesseract_confidence": (
                    None if tess_conf is None else float(tess_conf)
                ),
            }
        )
    return rows


def sweep(rows: list[dict]) -> list[dict]:
    """Residual CER vs k for instrument / random / layout / Tesseract."""
    cers = [r["sarvam_cer"] for r in rows]
    inst = [r["instrument_confidence"] for r in rows]
    layout = [float(r["layout_n_graphemes"]) for r in rows]
    tess_ok = all(r["tesseract_confidence"] is not None for r in rows)
    tess = [float(r["tesseract_confidence"] or 0.0) for r in rows]
    n = len(rows)
    table = []
    for k in range(0, n + 1):
        frac = k / n if n else float("nan")
        table.append(
            {
                "k": k,
                "frac_escalated": frac,
                "instrument": residual_system_cer(
                    cers, escalate_lowest_scores(inst, k)
                ),
                "random": random_mean_residual(cers, k),
                "layout_grapheme_count": residual_system_cer(
                    cers, escalate_highest_scores(layout, k)
                ),
                "tesseract": (
                    residual_system_cer(cers, escalate_lowest_scores(tess, k))
                    if tess_ok
                    else float("nan")
                ),
            }
        )
    return table


def write_report(repo_root: Path, out_path: Path, *, compute: bool) -> str:
    """Write docs/stage6_triage_cascade.md (protocol, or results if compute)."""
    lines = [
        "# Stage 6 — triage cascade\n",
        "Depends on Stage 5b's Sarvam cache/jsonl (Decision #19: no new API).",
        "Router quality, not cost savings (Decision #16).\n",
        "## Protocol\n",
        "- **Kept-page error:** grapheme CER of Sarvam Extract vs GT.",
        "- **Escalated pages:** treated as CER = 0 (perfect human review).",
        "- **System error at k:** mean CER over all n pages after filling",
        "  zeros on the k escalated pages. Same k for every policy.",
        "- **Instrument policy:** escalate the k *lowest* Probe 5b",
        "  `mean_confidence` pages (3-seed mean, Hindi condition).",
        "- **Random:** mean system error over 1000 draws of k pages",
        f"  (RNG seed {RANDOM_SEED}).",
        "- **Layout:** escalate the k *longest* GT grapheme strings.",
        "  (Line crops have no Stage 3 layout-bucket labels.)",
        "- **Tesseract:** escalate the k lowest Tesseract confidences",
        "  on the matching `*_plain.png`.",
        "- **Fair slice reported in the headline:** k = round(0.2 × n),",
        "  plus the full k = 0…n table.",
        "",
    ]
    if not compute:
        lines += [
            "## Results\n",
            "**Not computed.** Stage 5b ρ has not been authorised yet;",
            "this cascade would peek at the same Sarvam CERs. Run after",
            "Stage 5b `--compute-now`.\n",
        ]
        text = "\n".join(lines)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return text

    rows = build_cascade_rows(repo_root)
    n = len(rows)
    if n < 3:
        lines += [
            "## Results\n",
            f"**Blocked:** only n={n} Hindi plains with Sarvam text +",
            "instrument confidence. Finish Stage 5b fetch or accept the",
            "cache-only n=10 cohort and re-run.\n",
        ]
        text = "\n".join(lines)
        out_path.write_text(text, encoding="utf-8")
        return text

    table = sweep(rows)
    k20 = int(round(0.2 * n))
    row20 = table[k20]
    lines += [
        "## Results\n",
        f"n = **{n}** Hindi plains with Sarvam CER.\n",
        f"### Matched escalation k = {k20} "
        f"({row20['frac_escalated']*100:.1f}% of pages)\n",
        "| Policy | Residual system CER |",
        "|---|---:|",
        f"| Instrument confidence | {row20['instrument']:.4f} |",
        f"| Random (mean of {N_RANDOM_DRAWS}) | {row20['random']:.4f} |",
        f"| Layout (GT grapheme count) | {row20['layout_grapheme_count']:.4f} |",
        f"| Tesseract confidence | {row20['tesseract']:.4f} |",
        "",
        "Lower residual CER is better (more of the true errors were escalated).",
        "",
        "### Full sweep\n",
        "| k | frac | instrument | random | layout | tesseract |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['k']} | {r['frac_escalated']:.3f} | "
            f"{r['instrument']:.4f} | {r['random']:.4f} | "
            f"{r['layout_grapheme_count']:.4f} | {r['tesseract']:.4f} |"
        )
    lines.append("")
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 6 cascade. No API. Default is protocol-only."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/stage6_triage_cascade.md")
    )
    parser.add_argument(
        "--compute-now",
        action="store_true",
        help="Write residual-CER table. Use only after Stage 5b is authorised.",
    )
    args = parser.parse_args(argv)
    write_report(args.repo_root, args.out, compute=args.compute_now)
    print(f"Wrote {args.out} (compute={args.compute_now})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Adjudication sample + bootstrap Tier-1 rate + engine/language ranking.

Why this exists: docs/results_analysis.md / README cite ~20.4% of
Tesseract non-exact rows as Tier 1 encoding variants, but a large
slice of error_taxonomy.csv is UNREVIEWED. That makes the headline
provisional. This module:

1. Draws a stratified random sample of UNREVIEWED rows (fixed seed)
   into a notes-schema jsonl that hand_review.py --queue can label
   with the existing interactive tooling — no parallel review UI.
2. Once labels exist, bootstrap-estimates the Tier 1 (encoding-variant)
   rate over the FULL non-exact population, imputing unsampled
   UNREVIEWED from the adjudicated sample's label distribution, and
   prints the denominator at every step.
3. Tests whether Tier 1 normalisation reorders engines or languages
   by error rate — a ranking change is a stronger abstract claim than
   "X% of diffs aren't errors."

Do not fabricate labels. `sample` writes the queue; humans label;
`bootstrap` / `ranking` read whatever notes exist.

Called from: CLI after error_taxonomy.csv exists.
Writes: data/predictions/adjudication_sample.jsonl (default),
        docs/adjudication_analysis.md (optional --out for reports).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# Reuse Stage 0 loaders / classifiers without depending on analysis pkg.
_EVAL = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from equivalence_tables import normalize_tier1, normalize_whitespace  # noqa: E402
from hand_review_assist import SUGGESTED_LABELS  # noqa: E402

DEFAULT_TAXONOMY = Path("data/predictions/error_taxonomy.csv")
DEFAULT_NOTES = Path("data/predictions/hand_review_notes.jsonl")
DEFAULT_SAMPLE = Path("data/predictions/adjudication_sample.jsonl")
DEFAULT_PRED_ROOT = Path("data/predictions")
DEFAULT_RAW_ROOT = Path("data/raw")

SAMPLE_SEED = 42
SAMPLE_N = 200
BOOT_N = 10_000
BOOT_SEED = 42

# Human-typed labels that count toward the Tier-1 / "not a real error"
# numerator when adjudicating UNREVIEWED (automated Tier 1 already
# failed on these rows — this catches table gaps the reviewer finds).
ENCODING_VARIANT_LABELS = frozenset({
    "encoding-variant",
    "tier1",
    "tier-1",
    "exact-match",
    "exact",
    "not-an-error",
})

GENUINE_LABELS = frozenset(SUGGESTED_LABELS)

ENGINES = ("tesseract", "surya", "paddleocr")
LANGUAGES = ("hindi", "bengali", "santhali", "kashmiri")


def load_taxonomy(path: Path) -> list[dict[str, str]]:
    """Load error_taxonomy.csv (non-exact rows only, by construction)."""
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Stable identity for taxonomy / notes / sample join."""
    return (
        str(row["engine"]),
        str(row["language"]),
        str(row["image_id"]),
        str(row["variant"]),
    )


def load_notes(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """
    Latest note per (engine, language, image_id, variant).

    Append-only notes files can have multiple rows for the same key;
    the last non-skipped label wins for adjudication joins.
    """
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            note = json.loads(line)
            key = (
                note.get("engine", "tesseract"),
                note["language"],
                note["image_id"],
                note["variant"],
            )
            # Keep last write; bootstrap ignores pending/skipped.
            index[key] = note
    return index


def stratified_sample(
    unreviewed: list[dict[str, str]],
    n: int = SAMPLE_N,
    seed: int = SAMPLE_SEED,
) -> list[dict[str, str]]:
    """
    Sample up to n UNREVIEWED rows, stratified by (engine, language).

    Proportional allocation with at least one draw per non-empty
    stratum when n >= n_strata; leftover seats go to the largest
    remainders (Hamilton). Within a stratum, sample without
    replacement via a seeded RNG. If n >= population, return a
    shuffled copy of all UNREVIEWED (still seeded for reproducibility).
    """
    if not unreviewed:
        return []
    rng = np.random.default_rng(seed)
    if n >= len(unreviewed):
        order = rng.permutation(len(unreviewed))
        return [unreviewed[i] for i in order]

    by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in unreviewed:
        by_stratum[(row["engine"], row["language"])].append(row)

    strata = sorted(by_stratum.keys())
    pop = len(unreviewed)
    # Ideal quotas; guarantee ≥1 when possible.
    raw = {s: n * len(by_stratum[s]) / pop for s in strata}
    alloc = {s: max(1, int(math.floor(raw[s]))) for s in strata}
    # Cap at stratum size.
    for s in strata:
        alloc[s] = min(alloc[s], len(by_stratum[s]))
    total = sum(alloc.values())
    # If over-allocated from the ≥1 floor, trim largest strata.
    while total > n:
        s = max(strata, key=lambda k: (alloc[k], raw[k]))
        if alloc[s] <= 1 and all(alloc[k] <= 1 for k in strata):
            break
        if alloc[s] > 1:
            alloc[s] -= 1
            total -= 1
        else:
            break
    # Distribute remainders by largest fractional part.
    while total < n:
        candidates = [
            s for s in strata if alloc[s] < len(by_stratum[s])
        ]
        if not candidates:
            break
        s = max(candidates, key=lambda k: (raw[k] - math.floor(raw[k]), raw[k]))
        alloc[s] += 1
        total += 1

    sampled: list[dict[str, str]] = []
    for s in strata:
        pool = by_stratum[s]
        k = alloc[s]
        idx = rng.choice(len(pool), size=k, replace=False)
        for i in idx:
            sampled.append(pool[int(i)])
    # Final shuffle so review order is not stratum-blocked.
    order = rng.permutation(len(sampled))
    return [sampled[i] for i in order]


def to_notes_schema(
    row: dict[str, str],
    *,
    sample_seed: int,
    sample_index: int,
) -> dict[str, Any]:
    """
    One pending adjudication row in hand_review_notes.jsonl schema.

    label is null; suggestion_outcome is pending-adjudication so
    error_taxonomy.py (which only reads label) ignores these until a
    human session writes a real note to NOTES_PATH. hand_review.py
    --queue reads this file and appends completed notes there.
    """
    return {
        "language": row["language"],
        "image_id": row["image_id"],
        "variant": row["variant"],
        "engine": row["engine"],
        "ground_truth": row["ground_truth"],
        "predicted": row["predicted"],
        "label": None,
        "suggested_label": None,
        "suggested_reason": "pending adjudication sample — label via hand_review.py --queue",
        "suggestion_fits": False,
        "suggestion_outcome": "pending-adjudication",
        "adjudication_sample": True,
        "sample_seed": sample_seed,
        "sample_index": sample_index,
        "taxonomy_bucket_at_sample": row.get("bucket", "UNREVIEWED"),
    }


def write_sample(
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    out_path: Path = DEFAULT_SAMPLE,
    n: int = SAMPLE_N,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Draw sample, write notes-schema jsonl, return summary counts."""
    rows = load_taxonomy(taxonomy_path)
    unreviewed = [r for r in rows if r["bucket"] == "UNREVIEWED"]
    sampled = stratified_sample(unreviewed, n=n, seed=seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(sampled):
            f.write(
                json.dumps(
                    to_notes_schema(row, sample_seed=seed, sample_index=i),
                    ensure_ascii=False,
                )
                + "\n"
            )

    stratum_counts: dict[str, int] = defaultdict(int)
    for row in sampled:
        stratum_counts[f"{row['engine']}/{row['language']}"] += 1

    summary = {
        "taxonomy_path": taxonomy_path.as_posix(),
        "out_path": out_path.as_posix(),
        "sample_seed": seed,
        "requested_n": n,
        "n_taxonomy_non_exact": len(rows),
        "n_unreviewed_population": len(unreviewed),
        "n_sampled": len(sampled),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "denominator_note": (
            "Sample is drawn from UNREVIEWED rows only "
            f"(n_unreviewed={len(unreviewed)}). "
            f"Taxonomy non-exact denominator={len(rows)}."
        ),
    }
    return summary


def _is_encoding_variant_label(label: str | None) -> bool:
    if not label:
        return False
    return label.strip().lower() in ENCODING_VARIANT_LABELS


def _note_is_adjudicated(note: dict[str, Any]) -> bool:
    """True when a human session produced a usable label (not pending/skip)."""
    if note.get("suggestion_outcome") in {"pending-adjudication", "skipped"}:
        return False
    return note.get("label") is not None


def bootstrap_tier1_rate(
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    notes_path: Path = DEFAULT_NOTES,
    sample_path: Path = DEFAULT_SAMPLE,
    n_boot: int = BOOT_N,
    boot_seed: int = BOOT_SEED,
    engine: str | None = None,
) -> dict[str, Any]:
    """
    Bootstrap the encoding-variant (Tier 1) rate among non-exact rows.

    Population = taxonomy non-exact rows (optionally filtered to one
    engine). Each row contributes:
      - bucket TIER1 / TIER2 → known encoding/phonetic (Tier1 numerator
        only counts TIER1; TIER2 reported separately)
      - bucket GENUINE → known not Tier 1
      - bucket UNREVIEWED + adjudicated note → use human label
        (encoding-variant labels count as Tier 1)
      - bucket UNREVIEWED + not adjudicated → impute each bootstrap
        draw from the empirical label distribution of the *adjudicated
        sample* (and only those sample rows that have labels)

    Denominators are printed explicitly: this is the whole point of
    the repair relative to the bare 20.4% headline.
    """
    rows = load_taxonomy(taxonomy_path)
    if engine:
        rows = [r for r in rows if r["engine"] == engine]
    notes = load_notes(notes_path)

    sample_keys: set[tuple[str, str, str, str]] = set()
    if sample_path.is_file():
        with sample_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_keys.add(row_key(json.loads(line)))

    n_total = len(rows)
    n_tier1 = sum(1 for r in rows if r["bucket"] == "TIER1")
    n_tier2 = sum(1 for r in rows if r["bucket"] == "TIER2")
    n_genuine = sum(1 for r in rows if r["bucket"] == "GENUINE")
    n_unreviewed = sum(1 for r in rows if r["bucket"] == "UNREVIEWED")

    # Adjudicated sample labels among UNREVIEWED that were sampled.
    sample_labels: list[str] = []
    n_sample_adjudicated = 0
    n_sample_encoding = 0
    n_sample_pending = 0
    for r in rows:
        if r["bucket"] != "UNREVIEWED":
            continue
        key = row_key(r)
        if key not in sample_keys:
            continue
        note = notes.get(key)
        if note is None or not _note_is_adjudicated(note):
            n_sample_pending += 1
            continue
        n_sample_adjudicated += 1
        lab = str(note["label"])
        sample_labels.append(lab)
        if _is_encoding_variant_label(lab):
            n_sample_encoding += 1

    naive_rate = n_tier1 / n_total if n_total else float("nan")

    result: dict[str, Any] = {
        "engine_filter": engine or "ALL",
        "denominator_non_exact": n_total,
        "n_tier1_auto": n_tier1,
        "n_tier2_auto": n_tier2,
        "n_genuine_labeled": n_genuine,
        "n_unreviewed": n_unreviewed,
        "n_sample_keys": len(sample_keys),
        "n_sample_adjudicated": n_sample_adjudicated,
        "n_sample_pending": n_sample_pending,
        "n_sample_encoding_variant_labels": n_sample_encoding,
        "naive_tier1_rate_auto_only": naive_rate,
        "ready_for_bootstrap": n_sample_adjudicated > 0,
    }

    if n_sample_adjudicated == 0:
        result["note"] = (
            "No adjudicated sample labels yet. Run "
            "`hand_review.py --queue data/predictions/adjudication_sample.jsonl` "
            "then re-run bootstrap. Naive auto-only Tier1 rate is reported "
            "above; CI withheld."
        )
        result["tier1_rate_point"] = None
        result["tier1_rate_ci95"] = None
        return result

    # Empirical P(encoding | adjudicated sample).
    p_enc = n_sample_encoding / n_sample_adjudicated
    # Point estimate: auto Tier1 + expected encoding from all UNREVIEWED
    # using sample rate (Horvitz–Thompson style on the UNREVIEWED pool).
    point = (n_tier1 + n_unreviewed * p_enc) / n_total
    result["p_encoding_in_adjudicated_sample"] = p_enc
    result["tier1_rate_point"] = point

    # Build per-row status for bootstrap.
    # status: "tier1" | "not" | "impute"
    statuses: list[str] = []
    for r in rows:
        b = r["bucket"]
        if b == "TIER1":
            statuses.append("tier1")
        elif b in {"TIER2", "GENUINE"}:
            statuses.append("not")
        else:
            key = row_key(r)
            note = notes.get(key)
            if note is not None and _note_is_adjudicated(note):
                statuses.append(
                    "tier1" if _is_encoding_variant_label(str(note["label"])) else "not"
                )
            else:
                statuses.append("impute")

    rng = np.random.default_rng(boot_seed)
    statuses_arr = np.asarray(statuses)
    n = len(statuses_arr)
    rates = np.empty(n_boot, dtype=float)
    # Precompute sample Bernoulli for imputation.
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        drawn = statuses_arr[idx]
        n_t1 = int(np.sum(drawn == "tier1"))
        n_imp = int(np.sum(drawn == "impute"))
        if n_imp:
            n_t1 += int(rng.binomial(n_imp, p_enc))
        rates[b] = n_t1 / n

    lo, hi = np.percentile(rates, [2.5, 97.5])
    result["tier1_rate_ci95"] = [float(lo), float(hi)]
    result["tier1_rate_boot_mean"] = float(rates.mean())
    result["tier1_rate_boot_sd"] = float(rates.std(ddof=1))
    result["n_boot"] = n_boot
    result["note"] = (
        "Point estimate = (n_tier1_auto + n_unreviewed * "
        "p_encoding_in_sample) / denominator_non_exact. "
        "Bootstrap resamples non-exact rows; imputes unsampled "
        "UNREVIEWED from the adjudicated-sample encoding rate. "
        f"Adjudicated sample denominator={n_sample_adjudicated}."
    )
    return result


def _load_gt(language: str, raw_root: Path) -> dict[str, dict]:
    path = raw_root / language / "ground_truth.jsonl"
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _load_preds(engine: str, language: str, pred_root: Path) -> list[dict]:
    path = pred_root / engine / f"{language}.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ranking_test(
    pred_root: Path = DEFAULT_PRED_ROOT,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> dict[str, Any]:
    """
    Compare engine and language orderings by error rate before vs after
    Tier 1 normalisation.

    Raw error: whitespace-normalized strings differ.
    Tier1-aware error: normalize_tier1(gt) != normalize_tier1(pred).
    (Tier 1 includes whitespace via normalize_tier1's pipeline.)

    A reorder of engines or languages is the abstract-ready claim;
    a stable order means Tier 1 shrinks rates without changing ranks.
    """
    # Per engine aggregate and per language aggregate.
    engine_stats: dict[str, dict[str, int]] = {
        e: {"n": 0, "raw_err": 0, "tier1_err": 0} for e in ENGINES
    }
    lang_stats: dict[str, dict[str, int]] = {
        lang: {"n": 0, "raw_err": 0, "tier1_err": 0} for lang in LANGUAGES
    }
    # Also engine×language for the table.
    cell: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "raw_err": 0, "tier1_err": 0}
    )

    for language in LANGUAGES:
        gt_index = _load_gt(language, raw_root)
        if not gt_index:
            continue
        for engine in ENGINES:
            preds = _load_preds(engine, language, pred_root)
            for pred in preds:
                if pred.get("skipped_reason"):
                    continue
                gt = gt_index.get(pred["id"])
                if gt is None:
                    continue
                gt_text = gt["text"]
                hyp = pred.get("predicted_text") or ""
                raw_bad = normalize_whitespace(gt_text) != normalize_whitespace(hyp)
                tier1_bad = normalize_tier1(gt_text) != normalize_tier1(hyp)

                engine_stats[engine]["n"] += 1
                lang_stats[language]["n"] += 1
                cell[(engine, language)]["n"] += 1
                if raw_bad:
                    engine_stats[engine]["raw_err"] += 1
                    lang_stats[language]["raw_err"] += 1
                    cell[(engine, language)]["raw_err"] += 1
                if tier1_bad:
                    engine_stats[engine]["tier1_err"] += 1
                    lang_stats[language]["tier1_err"] += 1
                    cell[(engine, language)]["tier1_err"] += 1

    def rate_table(stats: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        rows = []
        for name, s in stats.items():
            if s["n"] == 0:
                continue
            rows.append({
                "name": name,
                "n": s["n"],
                "raw_error_rate": s["raw_err"] / s["n"],
                "tier1_error_rate": s["tier1_err"] / s["n"],
                "n_raw_err": s["raw_err"],
                "n_tier1_err": s["tier1_err"],
            })
        return rows

    engines = rate_table(engine_stats)
    languages = rate_table(lang_stats)

    def order_by(rows: list[dict[str, Any]], key: str) -> list[str]:
        return [r["name"] for r in sorted(rows, key=lambda x: x[key])]

    eng_raw = order_by(engines, "raw_error_rate")
    eng_t1 = order_by(engines, "tier1_error_rate")
    lang_raw = order_by(languages, "raw_error_rate")
    lang_t1 = order_by(languages, "tier1_error_rate")

    return {
        "engines": engines,
        "languages": languages,
        "engine_order_raw_best_to_worst": eng_raw,
        "engine_order_tier1_best_to_worst": eng_t1,
        "engine_order_changed": eng_raw != eng_t1,
        "language_order_raw_best_to_worst": lang_raw,
        "language_order_tier1_best_to_worst": lang_t1,
        "language_order_changed": lang_raw != lang_t1,
        "cells": {
            f"{e}/{lang}": cell[(e, lang)]
            for e, lang in cell
            if cell[(e, lang)]["n"]
        },
        "denominator_note": (
            "Rates use all non-skipped predictions with GT "
            "(not the non-exact-only taxonomy CSV). "
            "raw_error = whitespace-normalized mismatch; "
            "tier1_error = normalize_tier1 mismatch."
        ),
    }


def render_report(
    sample_summary: dict[str, Any] | None,
    boot_by_engine: dict[str, dict[str, Any]],
    ranking: dict[str, Any],
) -> str:
    """Markdown report; honest about pending adjudication."""
    lines = [
        "# Adjudication sample + Tier 1 rate + ranking",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        "**Code:** `src/analysis/adjudication_sample.py`  ",
        "**Labels:** not fabricated — sample queue for humans; "
        "bootstrap CI appears only after `hand_review.py --queue`.",
        "",
        "---",
        "",
    ]
    if sample_summary:
        lines.extend([
            "## 1. Sample",
            "",
            f"- UNREVIEWED population: **{sample_summary['n_unreviewed_population']}**",
            f"- Sampled: **{sample_summary['n_sampled']}** "
            f"(requested {sample_summary['requested_n']}, "
            f"seed={sample_summary['sample_seed']})",
            f"- Taxonomy non-exact denominator: "
            f"**{sample_summary['n_taxonomy_non_exact']}**",
            f"- Output: `{sample_summary['out_path']}`",
            "",
            "| Stratum (engine/language) | n sampled |",
            "|---------------------------|----------:|",
        ])
        for k, v in sample_summary["stratum_counts"].items():
            lines.append(f"| {k} | {v} |")
        lines.extend([
            "",
            "Label with:",
            "```bash",
            "PYTHONPATH=src/eval python3 src/eval/hand_review.py "
            "--queue data/predictions/adjudication_sample.jsonl",
            "```",
            "",
            "Typed overrides recognized as encoding-variant (count toward "
            "Tier 1 in bootstrap): "
            + ", ".join(sorted(ENCODING_VARIANT_LABELS))
            + ". Residual labels stay the Stage 0 four "
            + ", ".join(SUGGESTED_LABELS) + ".",
            "",
        ])

    lines.extend(["## 2. Bootstrap Tier 1 rate (non-exact denominator)", ""])
    for eng, boot in boot_by_engine.items():
        lines.append(f"### {eng}")
        lines.append("")
        lines.append(f"| Quantity | Value |")
        lines.append(f"|----------|------:|")
        for key in (
            "denominator_non_exact",
            "n_tier1_auto",
            "n_tier2_auto",
            "n_genuine_labeled",
            "n_unreviewed",
            "n_sample_adjudicated",
            "n_sample_pending",
            "n_sample_encoding_variant_labels",
            "naive_tier1_rate_auto_only",
            "p_encoding_in_adjudicated_sample",
            "tier1_rate_point",
        ):
            if key in boot and boot[key] is not None:
                val = boot[key]
                if isinstance(val, float):
                    lines.append(f"| {key} | {val:.4f} |")
                else:
                    lines.append(f"| {key} | {val} |")
        ci = boot.get("tier1_rate_ci95")
        if ci:
            lines.append(f"| tier1_rate_ci95 | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        else:
            lines.append("| tier1_rate_ci95 | *withheld — no adjudicated labels* |")
        lines.append("")
        if boot.get("note"):
            lines.append(boot["note"])
            lines.append("")

    lines.extend([
        "## 3. Ranking test (Tier 1 normalisation)",
        "",
        ranking["denominator_note"],
        "",
        "### Engines (lower error rate = better)",
        "",
        "| Engine | n | Raw error rate | Tier1 error rate |",
        "|--------|---|---------------:|-----------------:|",
    ])
    for r in sorted(ranking["engines"], key=lambda x: x["raw_error_rate"]):
        lines.append(
            f"| {r['name']} | {r['n']} | {r['raw_error_rate']:.4f} | "
            f"{r['tier1_error_rate']:.4f} |"
        )
    lines.extend([
        "",
        f"- Order raw (best→worst): "
        f"{ranking['engine_order_raw_best_to_worst']}",
        f"- Order after Tier 1: "
        f"{ranking['engine_order_tier1_best_to_worst']}",
        f"- **Order changed:** "
        f"{'YES' if ranking['engine_order_changed'] else 'no'}",
        "",
        "### Languages (lower error rate = better)",
        "",
        "| Language | n | Raw error rate | Tier1 error rate |",
        "|----------|---|---------------:|-----------------:|",
    ])
    for r in sorted(ranking["languages"], key=lambda x: x["raw_error_rate"]):
        lines.append(
            f"| {r['name']} | {r['n']} | {r['raw_error_rate']:.4f} | "
            f"{r['tier1_error_rate']:.4f} |"
        )
    lines.extend([
        "",
        f"- Order raw (best→worst): "
        f"{ranking['language_order_raw_best_to_worst']}",
        f"- Order after Tier 1: "
        f"{ranking['language_order_tier1_best_to_worst']}",
        f"- **Order changed:** "
        f"{'YES' if ranking['language_order_changed'] else 'no'}",
        "",
    ])
    if ranking["engine_order_changed"] or ranking["language_order_changed"]:
        lines.append(
            "**Abstract-ready:** Tier 1 normalisation reorders "
            + (
                "engines"
                if ranking["engine_order_changed"]
                else ""
            )
            + (
                " and "
                if ranking["engine_order_changed"] and ranking["language_order_changed"]
                else ""
            )
            + (
                "languages"
                if ranking["language_order_changed"]
                else ""
            )
            + " by error rate — stronger than a single “X% aren’t errors” rate."
        )
    else:
        lines.append(
            "**Ranking stable:** Tier 1 shrinks error rates but does not "
            "reorder engines or languages on this corpus. The 20.4%-style "
            "rate claim stays a magnitude claim, not a leaderboard claim."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="UNREVIEWED adjudication sample, bootstrap, ranking"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Draw stratified UNREVIEWED sample")
    p_sample.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p_sample.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    p_sample.add_argument("--n", type=int, default=SAMPLE_N)
    p_sample.add_argument("--seed", type=int, default=SAMPLE_SEED)

    p_boot = sub.add_parser(
        "bootstrap", help="Bootstrap Tier1 rate (needs adjudicated labels)"
    )
    p_boot.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p_boot.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    p_boot.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p_boot.add_argument("--n-boot", type=int, default=BOOT_N)
    p_boot.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    p_boot.add_argument(
        "--engine",
        default=None,
        help="Restrict to one engine (default: run ALL + each engine)",
    )

    p_rank = sub.add_parser("ranking", help="Engine/language order raw vs Tier1")
    p_rank.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    p_rank.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)

    p_all = sub.add_parser(
        "all", help="sample + bootstrap + ranking + write markdown report"
    )
    p_all.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p_all.add_argument("--sample-out", type=Path, default=DEFAULT_SAMPLE)
    p_all.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    p_all.add_argument("--n", type=int, default=SAMPLE_N)
    p_all.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p_all.add_argument("--n-boot", type=int, default=BOOT_N)
    p_all.add_argument("--report", type=Path, default=Path("docs/adjudication_analysis.md"))
    p_all.add_argument(
        "--skip-sample",
        action="store_true",
        help="Do not rewrite the sample file (use existing)",
    )

    args = ap.parse_args()

    if args.cmd == "sample":
        summary = write_sample(args.taxonomy, args.out, args.n, args.seed)
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "bootstrap":
        engines = [args.engine] if args.engine else [None, *ENGINES]
        for eng in engines:
            boot = bootstrap_tier1_rate(
                args.taxonomy,
                args.notes,
                args.sample,
                n_boot=args.n_boot,
                boot_seed=args.boot_seed,
                engine=eng,
            )
            print(json.dumps(boot, indent=2))
            print()
        return

    if args.cmd == "ranking":
        ranking = ranking_test(args.pred_root, args.raw_root)
        print(json.dumps(ranking, indent=2, default=str))
        return

    if args.cmd == "all":
        if args.skip_sample and args.sample_out.is_file():
            sample_summary = {
                "taxonomy_path": args.taxonomy.as_posix(),
                "out_path": args.sample_out.as_posix(),
                "sample_seed": args.seed,
                "requested_n": args.n,
                "n_taxonomy_non_exact": len(load_taxonomy(args.taxonomy)),
                "n_unreviewed_population": sum(
                    1 for r in load_taxonomy(args.taxonomy)
                    if r["bucket"] == "UNREVIEWED"
                ),
                "n_sampled": sum(
                    1 for _ in args.sample_out.open(encoding="utf-8") if _.strip()
                ),
                "stratum_counts": {},
            }
            # Rebuild stratum counts from file.
            sc: dict[str, int] = defaultdict(int)
            with args.sample_out.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sc[f"{row['engine']}/{row['language']}"] += 1
            sample_summary["stratum_counts"] = dict(sorted(sc.items()))
        else:
            sample_summary = write_sample(
                args.taxonomy, args.sample_out, args.n, args.seed
            )
            print("[sample]", json.dumps(sample_summary))

        boot_by_engine = {
            "ALL": bootstrap_tier1_rate(
                args.taxonomy, args.notes, args.sample_out,
                n_boot=args.n_boot, engine=None,
            )
        }
        for eng in ENGINES:
            boot_by_engine[eng] = bootstrap_tier1_rate(
                args.taxonomy, args.notes, args.sample_out,
                n_boot=args.n_boot, engine=eng,
            )
        ranking = ranking_test()
        report = render_report(sample_summary, boot_by_engine, ranking)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"[report] wrote {args.report}")
        print(
            f"  engine_order_changed={ranking['engine_order_changed']}  "
            f"language_order_changed={ranking['language_order_changed']}"
        )
        for eng, boot in boot_by_engine.items():
            ci = boot.get("tier1_rate_ci95")
            print(
                f"  {eng}: naive={boot['naive_tier1_rate_auto_only']:.4f}  "
                f"adjudicated={boot['n_sample_adjudicated']}/"
                f"pending={boot['n_sample_pending']}  "
                f"ci={ci}"
            )


if __name__ == "__main__":
    main()

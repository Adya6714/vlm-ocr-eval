"""
Probe 2 analysis — confusion pairs and true-token mass under misreads.

Turns probe2_*_seed{N}.jsonl into the claim-facing table: top-15
confusion pairs with mean probability / rank of the CORRECT cluster
when the model picked something else. That number is the paper's
Probe 2 headline — either the encoder still carries usable signal
the argmax misses, or it does not. Report honestly either way.

Writes docs/probe2_confusion_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_PROBES_DIR = Path(__file__).resolve().parents[1] / "probes"
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe2_confusion_graph import (  # noqa: E402
    aggregate_confusion_pairs,
    qualitative_tag,
)


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discover_seed_paths(
    probe_dir: Path,
    script: str = "hindi",
    condition: str = "natural",
) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for seed in (0, 1, 2):
        p = probe_dir / f"probe2_{script}_{condition}_seed{seed}.jsonl"
        if p.is_file():
            found[seed] = p
    return found


def flatten_misreads(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        for m in r.get("misreads", []):
            out.append(m)
    return out


def analyze_one_seed(records: list[dict]) -> dict[str, Any]:
    misreads = flatten_misreads(records)
    pairs = aggregate_confusion_pairs(misreads)
    probs = [m["true_prob"] for m in misreads]
    ranks = [m["true_rank"] for m in misreads if m["true_rank"] is not None]
    in_top5 = [m["true_in_top5"] for m in misreads]
    # Mass bins for the "close-but-wrong vs completely off" read.
    arr = np.asarray(probs, dtype=float) if probs else np.asarray([])
    bins = {
        "p_lt_0.01": float(np.mean(arr < 0.01)) if len(arr) else None,
        "p_0.01_0.05": float(np.mean((arr >= 0.01) & (arr < 0.05))) if len(arr) else None,
        "p_0.05_0.20": float(np.mean((arr >= 0.05) & (arr < 0.20))) if len(arr) else None,
        "p_ge_0.20": float(np.mean(arr >= 0.20)) if len(arr) else None,
    }
    meta = {}
    if records:
        meta = {
            "checkpoint_script": records[0].get("checkpoint_script"),
            "training_condition": records[0].get("training_condition"),
            "seed": records[0].get("seed"),
        }
    return {
        "meta": meta,
        "n_images": len(records),
        "n_substitutions": len(misreads),
        "n_unique_pairs": len(pairs),
        "mean_true_prob": float(np.mean(probs)) if probs else None,
        "mean_true_rank": float(np.mean(ranks)) if ranks else None,
        "median_true_rank": float(np.median(ranks)) if ranks else None,
        "frac_true_in_top5": float(np.mean(in_top5)) if in_top5 else None,
        "mass_bins": bins,
        "top15": [
            {**row, "qualitative_tag": qualitative_tag(row["true_cluster"], row["predicted_cluster"])}
            for row in pairs[:15]
        ],
        "all_pairs": pairs,
    }


def aggregate_across_seeds(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if len(per_seed) < 2:
        return {
            "n_seeds": len(per_seed),
            "seeds": sorted(per_seed.keys()),
            "ready": False,
            "note": (
                f"Only seed(s) {sorted(per_seed.keys())} on disk; "
                "across-seed mean±SD withheld until seed{0,1,2} all exist."
            ),
        }
    seeds = sorted(per_seed.keys())

    def collect(key: str) -> dict[str, Any]:
        vals = []
        per = {}
        for s in seeds:
            v = per_seed[s].get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append(float(v))
            per[str(s)] = float(v)
        arr = np.asarray(vals, dtype=float)
        return {
            "per_seed": per,
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "sd": float(arr.std(ddof=1)) if len(arr) >= 2 else float("nan"),
        }

    # Pool misreads across seeds for a combined top-15 (counts add).
    pooled: list[dict] = []
    for s in seeds:
        for row in per_seed[s].get("all_pairs", []):
            # Re-expand roughly via count — re-aggregate from stored pairs
            # by repeating is wrong; better re-sum counts.
            pooled.append(row)

    # Re-aggregate pair counts across seeds.
    merged: dict[tuple[str, str], dict] = {}
    for row in pooled:
        key = (row["true_cluster"], row["predicted_cluster"])
        if key not in merged:
            merged[key] = {
                "true_cluster": row["true_cluster"],
                "predicted_cluster": row["predicted_cluster"],
                "count": 0,
                "prob_sum": 0.0,
                "rank_sum": 0.0,
                "rank_n": 0,
                "top5_sum": 0.0,
                "example_top5": row.get("example_top5"),
            }
        m = merged[key]
        c = row["count"]
        m["count"] += c
        if row.get("mean_true_prob") is not None:
            m["prob_sum"] += row["mean_true_prob"] * c
        if row.get("mean_true_rank") is not None:
            m["rank_sum"] += row["mean_true_rank"] * c
            m["rank_n"] += c
        if row.get("frac_true_in_top5") is not None:
            m["top5_sum"] += row["frac_true_in_top5"] * c

    combined = []
    for m in merged.values():
        c = m["count"]
        combined.append({
            "true_cluster": m["true_cluster"],
            "predicted_cluster": m["predicted_cluster"],
            "count": c,
            "mean_true_prob": m["prob_sum"] / c if c else None,
            "mean_true_rank": m["rank_sum"] / m["rank_n"] if m["rank_n"] else None,
            "frac_true_in_top5": m["top5_sum"] / c if c else None,
            "example_top5": m["example_top5"],
            "qualitative_tag": qualitative_tag(m["true_cluster"], m["predicted_cluster"]),
        })
    combined.sort(key=lambda r: (-r["count"], r["true_cluster"], r["predicted_cluster"]))

    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ready": True,
        "mean_true_prob": collect("mean_true_prob"),
        "mean_true_rank": collect("mean_true_rank"),
        "frac_true_in_top5": collect("frac_true_in_top5"),
        "n_substitutions": collect("n_substitutions"),
        "top15_pooled": combined[:15],
    }


def analyze_all_seeds(seed_paths: dict[int, Path]) -> dict[str, Any]:
    per_seed = {}
    for seed, path in sorted(seed_paths.items()):
        records = load_records(path)
        if not records:
            continue
        per_seed[seed] = analyze_one_seed(records)
        per_seed[seed]["source"] = path.as_posix()
    return {
        "per_seed": per_seed,
        "across_seeds": aggregate_across_seeds(per_seed),
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def render_markdown(bundle: dict[str, Any]) -> str:
    across = bundle["across_seeds"]
    seeds = sorted(bundle["per_seed"].keys())
    primary = bundle["per_seed"][seeds[0]]
    sources = [f"`{bundle['per_seed'][s].get('source', '')}`" for s in seeds]
    meta0 = primary["meta"]
    script = meta0.get("checkpoint_script", "hindi")
    condition = meta0.get("training_condition", "natural")

    lines = [
        "# Probe 2 — confusion structure (GT-aligned)",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Sources:** {', '.join(sources)}  ",
        f"**Run:** {script}/{condition}/seeds {seeds}  ",
        "**Method:** Needleman–Wunsch grapheme alignment; at each "
        "substitution record top-5 and full-softmax p(true)/rank "
        "(DECISIONS.md #57). Sample = Probe 5b Hindi Random(0) draw. "
        "Checkpoints: `checkpoint_{script}_{condition}_seed{N}.pt` (#47).",
        "",
        "---",
        "",
    ]

    if across.get("ready"):
        lines.extend([
            "## 1. Across-seed headline — mass on the correct cluster",
            "",
            "When the model is **wrong**, how much probability did it "
            "still assign to the true grapheme? High mass ⇒ encoder "
            "signal the argmax/confidence readout fails to surface. "
            "Near-zero / near-uniform ⇒ the opposite.",
            "",
            "| Quantity | " + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean | SD |",
            "|----------|" + "|".join(["------"] * len(seeds)) + "|------|----|",
        ])
        for label, key in [
            ("mean p(true) on misreads", "mean_true_prob"),
            ("mean rank(true)", "mean_true_rank"),
            ("frac true in top-5", "frac_true_in_top5"),
            ("n substitutions", "n_substitutions"),
        ]:
            block = across[key]
            seed_cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {seed_cols} | {_fmt(block['mean'])} | "
                f"{_fmt(block['sd'])} |"
            )
        lines.append("")

        lines.extend([
            "## 2. Top 15 confusion pairs (pooled counts across seeds)",
            "",
            "| # | n | true | predicted | mean p(true) | mean rank | "
            "in top-5 | tag |",
            "|---|---|------|-----------|--------------|-----------|"
            "---------|-----|",
        ])
        for i, row in enumerate(across["top15_pooled"], start=1):
            lines.append(
                f"| {i} | {row['count']} | `{row['true_cluster']}` | "
                f"`{row['predicted_cluster']}` | "
                f"{_fmt(row['mean_true_prob'])} | "
                f"{_fmt(row['mean_true_rank'], 1)} | "
                f"{_fmt(row['frac_true_in_top5'], 2)} | "
                f"{row['qualitative_tag']} |"
            )
        lines.append("")
        lines.extend([
            "### Qualitative read",
            "",
            "`same-base-matra-diff` / `adjacent-codepoint` / "
            "`same-base` suggest visually or structurally sensible "
            "confusions (matras, nearby aksaras). `dissimilar` / "
            "unrelated `both-devanagari` pairs look more like prior "
            "noise. Tags are heuristics — read the Unicode pairs.",
            "",
        ])
    else:
        lines.append(f"*{across.get('note')}*")
        lines.append("")

    lines.extend(["## 3. Per-seed detail", ""])
    for s in seeds:
        r = bundle["per_seed"][s]
        lines.append(f"### Seed {s}")
        lines.append("")
        lines.append(
            f"Source: `{r.get('source', '')}` — {r['n_images']} images, "
            f"{r['n_substitutions']} substitutions, "
            f"{r['n_unique_pairs']} unique pairs."
        )
        lines.append("")
        lines.append(
            f"mean p(true) = {_fmt(r['mean_true_prob'])}, "
            f"mean rank = {_fmt(r['mean_true_rank'], 1)}, "
            f"frac in top-5 = {_fmt(r['frac_true_in_top5'], 3)}."
        )
        bins = r["mass_bins"]
        lines.append(
            f"Mass bins: p<0.01 {_fmt(bins['p_lt_0.01'], 3)}; "
            f"0.01–0.05 {_fmt(bins['p_0.01_0.05'], 3)}; "
            f"0.05–0.20 {_fmt(bins['p_0.05_0.20'], 3)}; "
            f"≥0.20 {_fmt(bins['p_ge_0.20'], 3)}."
        )
        lines.append("")
        lines.append("| # | n | true → pred | p(true) | rank | tag |")
        lines.append("|---|---|-------------|---------|------|-----|")
        for i, row in enumerate(r["top15"], start=1):
            lines.append(
                f"| {i} | {row['count']} | "
                f"`{row['true_cluster']}` → `{row['predicted_cluster']}` | "
                f"{_fmt(row['mean_true_prob'])} | "
                f"{_fmt(row['mean_true_rank'], 1)} | "
                f"{row['qualitative_tag']} |"
            )
        lines.append("")

    lines.extend([
        "## 4. Finding (plain language)",
        "",
    ])
    if across.get("ready"):
        mp = across["mean_true_prob"]["mean"]
        lines.append(
            f"Across seeds {seeds}, when the model substitutes a wrong "
            f"grapheme it still assigns mean probability "
            f"**{_fmt(mp)}** to the correct cluster "
            f"(mean rank {_fmt(across['mean_true_rank']['mean'], 1)}; "
            f"in top-5 "
            f"{_fmt(across['frac_true_in_top5']['mean'], 3)} of the time)."
        )
        lines.append("")
        if mp is not None and not math.isnan(mp):
            if mp >= 0.05:
                lines.append(
                    "That is **non-trivial mass on the true answer** — "
                    "evidence the encoder carries usable signal that "
                    "greedy argmax / confidence readout does not surface."
                )
            else:
                lines.append(
                    "That mass is **near floor** — when wrong, the model "
                    "is typically not 'close but argmax-unlucky'; the "
                    "correct cluster is not sitting just under the "
                    "chosen token. Report this as the opposite of the "
                    "hopeful Probe 2 story."
                )
        lines.append("")
        # Tag histogram on pooled top-15
        tag_counts: dict[str, int] = defaultdict(int)
        for row in across["top15_pooled"]:
            tag_counts[row["qualitative_tag"]] += 1
        tag_str = ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items()))
        lines.append(f"Top-15 qualitative tags: {tag_str}.")
    else:
        lines.append("Across-seed finding withheld until seeds 0/1/2 exist.")
    lines.append("")
    lines.append(
        "**What this does not establish:** production-API confusion "
        "structure (instrument only); that high p(true) on misreads "
        "implies a fix by temperature/sampling (would need a decoding "
        "ablation); or that dissimilar pairs are 'random' rather than "
        "prior-driven."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Probe 2 confusion jsonl")
    ap.add_argument("--input", default=None)
    ap.add_argument("--probe-dir", default="data/probe_results")
    ap.add_argument("--script", default="hindi")
    ap.add_argument("--condition", default="natural")
    ap.add_argument("--out", default="docs/probe2_confusion_analysis.md")
    args = ap.parse_args()

    if args.input:
        records = load_records(Path(args.input))
        one = analyze_one_seed(records)
        one["source"] = Path(args.input).as_posix()
        seed_key = int(one["meta"].get("seed", 0))
        bundle = {
            "per_seed": {seed_key: one},
            "across_seeds": aggregate_across_seeds({seed_key: one}),
        }
    else:
        paths = discover_seed_paths(
            Path(args.probe_dir), args.script, args.condition,
        )
        if not paths:
            raise SystemExit(
                f"no probe2_{args.script}_{args.condition}_seed*.jsonl "
                f"in {args.probe_dir}"
            )
        bundle = analyze_all_seeds(paths)

    md = render_markdown(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[analyze_probe2] wrote {out} (seeds={sorted(bundle['per_seed'].keys())})")
    across = bundle["across_seeds"]
    if across.get("ready"):
        print(
            f"  mean p(true)={_fmt(across['mean_true_prob']['mean'])} ± "
            f"{_fmt(across['mean_true_prob']['sd'])}"
        )
    else:
        print(f"  {across.get('note')}")


if __name__ == "__main__":
    main()

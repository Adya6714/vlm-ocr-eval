"""
Attention-ablation analysis — Claim B mechanism numbers.

Turns attention_ablation_*_seed{N}.jsonl into the claim-facing summary:
does mean_confidence under full encoder memory differ from
mean_confidence under zeroed encoder memory? How large are per-step
KL(full || zero), top-1 agreement, and prior-sufficiency?

Reuses the Probe 5b cluster-bootstrap pattern (resample images,
n_boot=10000, seed=42) for the confidence contrast — do not reinvent
(DECISIONS.md #51 / #56; docs/statistical_repair.md).

Writes docs/attention_ablation_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# Reuse bootstrap / TOST helpers so the repair methodology stays one
# code path across probes. Insert this directory so sibling import
# works whether invoked as `python src/analysis/...` or via PYTHONPATH=src.
_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from analyze_probe5b import (  # noqa: E402
    BOOT_SEED,
    DEFAULT_N_BOOT,
    SEOI_DELTA,
    conf_stats,
    tost_equivalence,
)

# Local bootstrap for a single pair of arrays (full vs zero), matching
# cluster_bootstrap's image-resampling idea without the 4-condition grid.


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discover_seed_paths(
    pattern_dir: Path,
    script: str = "hindi",
    condition: str = "natural",
) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for seed in (0, 1, 2):
        p = pattern_dir / f"attention_ablation_{script}_{condition}_seed{seed}.jsonl"
        if p.is_file():
            found[seed] = p
    return found


def paired_cluster_bootstrap(
    full: np.ndarray,
    zero: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """
    Cluster bootstrap for paired full vs zero-memory mean confidence.

    Resamples IMAGE indices with replacement (same index applied to
    both arrays — the pair is the cluster). Mirrors analyze_probe5b's
    unit of resampling so Claim B correlational and mechanistic
    intervals share a methodology.
    """
    assert len(full) == len(zero)
    n = len(full)
    rng = np.random.default_rng(rng_seed)
    boot_full = np.empty(n_boot, dtype=float)
    boot_zero = np.empty(n_boot, dtype=float)
    boot_diff = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mf = float(full[idx].mean())
        mz = float(zero[idx].mean())
        boot_full[b] = mf
        boot_zero[b] = mz
        boot_diff[b] = mf - mz

    def pct_ci(arr: np.ndarray, level: float = 0.95) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0 * 100.0
        hi = (1.0 + level) / 2.0 * 100.0
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    lo95, hi95 = pct_ci(boot_diff, 0.95)
    lo90, hi90 = pct_ci(boot_diff, 0.90)
    return {
        "n_boot": n_boot,
        "rng_seed": rng_seed,
        "full_mean_ci95": pct_ci(boot_full, 0.95),
        "zero_mean_ci95": pct_ci(boot_zero, 0.95),
        "diff_boot_mean": float(boot_diff.mean()),
        "diff_ci95": (lo95, hi95),
        "diff_ci90": (lo90, hi90),
        "diff_boot_sd": float(boot_diff.std(ddof=1)),
        "p_lower": float(np.mean(boot_diff <= -SEOI_DELTA)),
        "p_upper": float(np.mean(boot_diff >= SEOI_DELTA)),
    }


def analyze_one_seed(
    records: list[dict],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Per-seed summary of confidence contrast + distribution metrics."""
    full = np.asarray(
        [r["mean_confidence_full"] for r in records if r.get("mean_confidence_full") is not None],
        dtype=float,
    )
    zero = np.asarray(
        [r["mean_confidence_zero"] for r in records if r.get("mean_confidence_zero") is not None],
        dtype=float,
    )
    # Pair by record order — both fields required on every complete row.
    paired_full, paired_zero = [], []
    for r in records:
        if r.get("mean_confidence_full") is None or r.get("mean_confidence_zero") is None:
            continue
        paired_full.append(r["mean_confidence_full"])
        paired_zero.append(r["mean_confidence_zero"])
    pf = np.asarray(paired_full, dtype=float)
    pz = np.asarray(paired_zero, dtype=float)

    deltas = pf - pz
    kls = np.asarray(
        [r["mean_kl_full_given_zero"] for r in records if r.get("mean_kl_full_given_zero") is not None],
        dtype=float,
    )
    agrees = np.asarray(
        [r["top1_agreement_rate"] for r in records if r.get("top1_agreement_rate") is not None],
        dtype=float,
    )
    prior = np.asarray(
        [r["mean_prior_sufficiency"] for r in records if r.get("mean_prior_sufficiency") is not None],
        dtype=float,
    )

    boot = paired_cluster_bootstrap(pf, pz, n_boot=n_boot, rng_seed=rng_seed)
    tost = tost_equivalence(
        boot["diff_ci90"][0],
        boot["diff_ci90"][1],
        boot["p_lower"],
        boot["p_upper"],
    )

    meta = {}
    if records:
        meta = {
            "checkpoint_script": records[0].get("checkpoint_script"),
            "training_condition": records[0].get("training_condition"),
            "seed": records[0].get("seed"),
            "kl_direction": records[0].get("kl_direction", "KL(full || zero)"),
            "prior_sufficiency_definition": records[0].get(
                "prior_sufficiency_definition",
                "sum_i min(p_full[i], p_zero[i])",
            ),
            "prefix_alignment": records[0].get("prefix_alignment"),
        }

    return {
        "meta": meta,
        "n_images": len(records),
        "n_paired": int(pf.size),
        "full": conf_stats(full),
        "zero": conf_stats(zero),
        "delta": conf_stats(deltas),
        "mean_kl": conf_stats(kls),
        "top1_agreement": conf_stats(agrees),
        "prior_sufficiency": conf_stats(prior),
        "bootstrap": boot,
        "tost_delta_equiv": tost,
        "n_boot": n_boot,
        "seoi_delta": SEOI_DELTA,
    }


def aggregate_across_seeds(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-seed mean±SD of per-seed scalars — never pooled images."""
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

    def collect(field: str, sub: str = "mean") -> dict[str, Any]:
        vals = []
        per = {}
        for s in seeds:
            v = per_seed[s][field][sub]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append(float(v))
            per[str(s)] = float(v)
        arr = np.asarray(vals, dtype=float)
        return {
            "per_seed": per,
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "sd": float(arr.std(ddof=1)) if len(arr) >= 2 else float("nan"),
            "n_seeds": len(arr),
        }

    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ready": True,
        "mean_confidence_full": collect("full"),
        "mean_confidence_zero": collect("zero"),
        "mean_confidence_delta": collect("delta"),
        "mean_kl": collect("mean_kl"),
        "top1_agreement": collect("top1_agreement"),
        "prior_sufficiency": collect("prior_sufficiency"),
    }


def analyze_all_seeds(
    seed_paths: dict[int, Path],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    per_seed = {}
    for seed, path in sorted(seed_paths.items()):
        records = load_records(path)
        if not records:
            continue
        per_seed[seed] = analyze_one_seed(
            records, n_boot=n_boot, rng_seed=rng_seed + seed,
        )
        per_seed[seed]["source"] = path.as_posix()
    return {
        "per_seed": per_seed,
        "across_seeds": aggregate_across_seeds(per_seed),
        "n_boot": n_boot,
        "seoi_delta": SEOI_DELTA,
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def render_markdown(bundle: dict[str, Any]) -> str:
    """Claim-facing write-up for docs/attention_ablation_analysis.md."""
    across = bundle["across_seeds"]
    seeds = sorted(bundle["per_seed"].keys())
    primary = bundle["per_seed"][seeds[0]]
    sources = [f"`{bundle['per_seed'][s].get('source', '')}`" for s in seeds]
    meta0 = primary["meta"]
    script = meta0.get("checkpoint_script", "hindi")
    condition = meta0.get("training_condition", "natural")
    n_total = sum(r["n_images"] for r in bundle["per_seed"].values())

    lines: list[str] = [
        "# Attention ablation — does confidence depend on the image?",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Sources:** {', '.join(sources)}  ",
        f"**Run:** {script}/{condition}/seeds {seeds}  ",
        f"**Records:** {n_total} images across {len(seeds)} seed(s)  ",
        f"**Method:** full encoder memory vs all-zeros encoder memory "
        f"(zeroed *before* `memory_projection`). Per-step "
        f"{meta0.get('kl_direction', 'KL(full || zero)')}, top-1 "
        f"agreement, and prior sufficiency "
        f"(`{meta0.get('prior_sufficiency_definition', 'sum min')}`) "
        f"under teacher-forced shared prefixes "
        f"(`{meta0.get('prefix_alignment', 'teacher_force')}`). "
        f"Headline contrast = independent greedy "
        f"`mean_confidence_full` vs `mean_confidence_zero`. "
        f"Cluster bootstrap of images (n_boot = {bundle['n_boot']}), "
        f"TOST δ = {bundle['seoi_delta']} (same repair as Probe 5b). "
        f"DECISIONS.md #56.",
        "",
        "---",
        "",
    ]

    if across.get("ready"):
        lines.extend([
            "## 1. Across-seed headline (primary)",
            "",
            "Per-seed means, then across-seed mean ± SD. If "
            "`mean_confidence_full` ≈ `mean_confidence_zero`, confidence "
            "is prior-dominated — stronger than the correlational Claim B "
            "finding that confidence fails to separate real/blank/unseen.",
            "",
            "| Quantity | " + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean | SD |",
            "|----------|" + "|".join(["------"] * len(seeds)) + "|------|----|",
        ])
        for label, key in [
            ("mean conf (full)", "mean_confidence_full"),
            ("mean conf (zero-memory)", "mean_confidence_zero"),
            ("Δ (full − zero)", "mean_confidence_delta"),
            ("mean KL(full∥zero)", "mean_kl"),
            ("top-1 agreement rate", "top1_agreement"),
            ("mean prior sufficiency", "prior_sufficiency"),
        ]:
            block = across[key]
            seed_cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {seed_cols} | {_fmt(block['mean'])} | "
                f"{_fmt(block['sd'])} |"
            )
        lines.append("")

        d = across["mean_confidence_delta"]
        lines.extend([
            "### 1a. Reading the confidence Δ",
            "",
            f"Across-seed mean Δ(full − zero) = **{_fmt(d['mean'])}** "
            f"(SD {_fmt(d['sd'])}). ",
        ])
        # Qualitative gate without inventing a new δ: compare |Δ| to
        # across-seed SD of full confidence and to SEOI 0.05.
        full_sd = across["mean_confidence_full"]["sd"]
        abs_d = abs(d["mean"]) if not math.isnan(d["mean"]) else float("nan")
        if not math.isnan(abs_d):
            if abs_d < SEOI_DELTA:
                lines.append(
                    f"|Δ| < δ = {SEOI_DELTA} (Probe 5b SEOI) — confidence "
                    "under zeroed memory is practically equivalent to "
                    "full-memory confidence on the abstention scale."
                )
            else:
                lines.append(
                    f"|Δ| ≥ δ = {SEOI_DELTA} — zeroing memory moves "
                    "confidence by a practically meaningful amount."
                )
            if not math.isnan(full_sd) and abs_d < full_sd:
                lines.append(
                    f" |Δ| also sits below the across-seed SD of full "
                    f"confidence ({_fmt(full_sd)}) — seed noise swamps "
                    "the ablation effect."
                )
        lines.append("")
    else:
        lines.append(f"*{across.get('note', 'Across-seed summary unavailable.')}*")
        lines.append("")

    lines.extend(["## 2. Per-seed detail", ""])
    for s in seeds:
        result = bundle["per_seed"][s]
        boot = result["bootstrap"]
        tost = result["tost_delta_equiv"]
        lines.append(f"### Seed {s}")
        lines.append("")
        lines.append(
            f"Source: `{result.get('source', '')}` — "
            f"{result['n_images']} images, {result['n_paired']} paired."
        )
        lines.append("")
        lines.append(
            "| Condition | n | Mean | Naive 95% CI |"
        )
        lines.append("|-----------|---|------|--------------|")
        for label, key in [("full memory", "full"), ("zero memory", "zero"), ("Δ (full−zero)", "delta")]:
            row = result[key]
            lines.append(
                f"| {label} | {row['n']} | {_fmt(row['mean'])} | "
                f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] |"
            )
        lines.append("")
        lines.append(
            f"Bootstrap Δ 95% CI: [{_fmt(boot['diff_ci95'][0])}, "
            f"{_fmt(boot['diff_ci95'][1])}]; 90% CI: "
            f"[{_fmt(boot['diff_ci90'][0])}, {_fmt(boot['diff_ci90'][1])}]. "
            f"TOST equivalence at δ={SEOI_DELTA}: "
            f"{'yes' if tost['equivalent'] else 'no'}."
        )
        lines.append("")
        lines.append(
            f"Distribution metrics (image-level means of per-step "
            f"stats under shared prefixes): "
            f"KL(full∥zero) = {_fmt(result['mean_kl']['mean'])} "
            f"(SD {_fmt(result['mean_kl']['sd'])}); "
            f"top-1 agreement = {_fmt(result['top1_agreement']['mean'])}; "
            f"prior sufficiency = {_fmt(result['prior_sufficiency']['mean'])}."
        )
        lines.append("")

    lines.extend([
        "## 3. Metric definitions",
        "",
        "- **KL direction:** KL(p_full || p_zero) — primary. "
        "KL(p_zero || p_full) is also stored per step in the jsonl.",
        "- **Top-1 agreement:** whether argmax(p_full) equals "
        "argmax(p_zero) at the same prefix.",
        "- **Prior sufficiency:** "
        "`sum_i min(p_full[i], p_zero[i])` = `1 − TV(p_full, p_zero)`. "
        "Exact full-support overlap; not top-K restricted.",
        "- **Prefix alignment:** zero-memory distributions for KL / "
        "agreement / prior-sufficiency are teacher-forced along the "
        "full-memory greedy token sequence. Independent zero-memory "
        "greedy supplies `mean_confidence_zero` only.",
        "",
        "## 4. Finding (plain language)",
        "",
    ])
    if across.get("ready"):
        d = across["mean_confidence_delta"]
        ps = across["prior_sufficiency"]
        ta = across["top1_agreement"]
        kl = across["mean_kl"]
        lines.append(
            f"Across seeds {seeds}, mean confidence with full memory is "
            f"{_fmt(across['mean_confidence_full']['mean'])} and with "
            f"zeroed encoder memory is "
            f"{_fmt(across['mean_confidence_zero']['mean'])} "
            f"(Δ = {_fmt(d['mean'])} ± {_fmt(d['sd'])}). "
            f"Shared-prefix top-1 agreement averages "
            f"{_fmt(ta['mean'])}; prior-sufficiency "
            f"{_fmt(ps['mean'])}; mean KL(full∥zero) "
            f"{_fmt(kl['mean'])}."
        )
        lines.append("")
        lines.append(
            "If Δ is tiny and prior sufficiency / top-1 agreement stay "
            "high, Claim B is not only correlational (confidence fails "
            "to separate image conditions) but mechanistic: the "
            "decoder's confidence barely depends on encoder features."
        )
    else:
        lines.append(
            "Across-seed finding withheld until seeds 0/1/2 all exist."
        )
    lines.append("")
    lines.append(
        "**What this does not establish:** that production OCR APIs "
        "share the same prior-dominated confidence; that zeroing "
        "memory is identical to blank-image inputs (blank still "
        "produces nonzero encoder features); or that attention weights "
        "themselves are diffuse (this ablates memory content, not "
        "attention maps — BOOK.md methodology upgrade #2's weight "
        "introspection remains separate)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze attention-ablation jsonl")
    ap.add_argument("--input", default=None, help="Single jsonl (overrides discovery)")
    ap.add_argument("--probe-dir", default="data/probe_results")
    ap.add_argument("--script", default="hindi")
    ap.add_argument("--condition", default="natural")
    ap.add_argument("--out", default="docs/attention_ablation_analysis.md")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    args = ap.parse_args()

    if args.input:
        records = load_records(Path(args.input))
        if not records:
            raise SystemExit(f"no records in {args.input}")
        one = analyze_one_seed(records, n_boot=args.n_boot, rng_seed=args.boot_seed)
        one["source"] = Path(args.input).as_posix()
        seed_key = int(one["meta"].get("seed", 0))
        bundle = {
            "per_seed": {seed_key: one},
            "across_seeds": aggregate_across_seeds({seed_key: one}),
            "n_boot": args.n_boot,
            "seoi_delta": SEOI_DELTA,
        }
    else:
        seed_paths = discover_seed_paths(
            Path(args.probe_dir), args.script, args.condition,
        )
        if not seed_paths:
            raise SystemExit(
                f"no attention_ablation_{args.script}_{args.condition}_seed*.jsonl "
                f"in {args.probe_dir}"
            )
        bundle = analyze_all_seeds(
            seed_paths, n_boot=args.n_boot, rng_seed=args.boot_seed,
        )

    md = render_markdown(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(
        f"[analyze_attention_ablation] wrote {out} "
        f"(seeds={sorted(bundle['per_seed'].keys())})"
    )
    across = bundle["across_seeds"]
    if across.get("ready"):
        for key in (
            "mean_confidence_full",
            "mean_confidence_zero",
            "mean_confidence_delta",
            "mean_kl",
            "top1_agreement",
            "prior_sufficiency",
        ):
            block = across[key]
            print(
                f"  {key}: {_fmt(block['mean'])} ± {_fmt(block['sd'])}"
            )
    else:
        print(f"  across-seeds: {across.get('note')}")


if __name__ == "__main__":
    main()

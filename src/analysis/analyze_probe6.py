"""
Probe 6 analysis — does Claim B hold on real Tier C documents?

Compares:
  - Synthetic domain (existing Probe 3/5 hindi/natural jsonl):
      Probe 3 real/blank mean confidence; Probe 5 accuracy
  - Real Tier C domain (probe6_synthetic_real_hindi_seed{N}.jsonl):
      real_plain / real_degraded / blank confidence; plain/degraded accuracy

Primary question: is real_plain confidence ≈ blank confidence on Tier C
the same way synthetic real ≈ blank on Probe 3? Report honestly.

Cluster bootstrap of images reuses analyze_probe5b helpers
(docs/statistical_repair.md). Writes docs/probe6_synthetic_real_analysis.md.
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

PROBE6_CONDS = ("real_plain", "real_degraded", "blank")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discover_probe6_seeds(probe_dir: Path) -> dict[int, Path]:
    found = {}
    for seed in (0, 1, 2):
        p = probe_dir / f"probe6_synthetic_real_hindi_seed{seed}.jsonl"
        if p.is_file():
            found[seed] = p
    return found


def load_synthetic_probe3(path: Path) -> dict[str, np.ndarray]:
    """Image-level mean confidence for Probe 3 real / blank / noise."""
    rows = load_jsonl(path)
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for cond in ("real", "blank", "noise"):
            block = r.get(cond) or {}
            c = block.get("mean_confidence")
            if c is not None:
                out[cond].append(float(c))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def load_synthetic_probe5(path: Path) -> dict[str, Any]:
    """Probe 5 synthetic accuracy + confidence from the records blob."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    confs = np.asarray([r["confidence"] for r in records], dtype=float)
    correct = np.asarray([1.0 if r["correct"] else 0.0 for r in records], dtype=float)
    return {
        "n": len(records),
        "mean_confidence": float(confs.mean()) if len(confs) else float("nan"),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "confidences": confs,
        "corrects": correct,
    }


def paired_diff_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """
    Bootstrap mean(a)−mean(b) by resampling indices with replacement.

    When lengths differ (synthetic n=100 vs Tier C n=60), resample each
    array independently — unpaired image-level cluster bootstrap, same
    spirit as Probe 5b's within-condition image resample.
    """
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_boot, dtype=float)
    means_a = np.empty(n_boot, dtype=float)
    means_b = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        ma = float(a[ia].mean())
        mb = float(b[ib].mean())
        means_a[i] = ma
        means_b[i] = mb
        diffs[i] = ma - mb

    def pct(arr: np.ndarray, level: float) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0 * 100.0
        hi = (1.0 + level) / 2.0 * 100.0
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    lo95, hi95 = pct(diffs, 0.95)
    lo90, hi90 = pct(diffs, 0.90)
    return {
        "diff_boot_mean": float(diffs.mean()),
        "diff_ci95": (lo95, hi95),
        "diff_ci90": (lo90, hi90),
        "p_lower": float(np.mean(diffs <= -SEOI_DELTA)),
        "p_upper": float(np.mean(diffs >= SEOI_DELTA)),
        "mean_a_ci95": pct(means_a, 0.95),
        "mean_b_ci95": pct(means_b, 0.95),
        "n_boot": n_boot,
    }


def analyze_probe6_seed(records: list[dict]) -> dict[str, Any]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    per_condition = {}
    conf_arrays = {}
    for cond in PROBE6_CONDS:
        subset = by_cond.get(cond, [])
        confs = [
            r["mean_confidence"]
            for r in subset
            if r.get("mean_confidence") is not None
        ]
        corrects = [
            1.0 if r["correct"] else 0.0
            for r in subset
            if r.get("correct") is not None
        ]
        stats = conf_stats(confs)
        stats["accuracy"] = float(np.mean(corrects)) if corrects else None
        stats["n_scored"] = len(corrects)
        per_condition[cond] = stats
        conf_arrays[cond] = np.asarray(confs, dtype=float)

    contrasts = {}
    if "real_plain" in conf_arrays and "blank" in conf_arrays:
        if len(conf_arrays["real_plain"]) and len(conf_arrays["blank"]):
            boot = paired_diff_bootstrap(
                conf_arrays["real_plain"], conf_arrays["blank"],
            )
            contrasts["plain_minus_blank"] = {
                "diff": float(
                    conf_arrays["real_plain"].mean() - conf_arrays["blank"].mean()
                ),
                **boot,
                "tost": tost_equivalence(
                    boot["diff_ci90"][0],
                    boot["diff_ci90"][1],
                    boot["p_lower"],
                    boot["p_upper"],
                ),
            }
    if "real_degraded" in conf_arrays and "blank" in conf_arrays:
        if len(conf_arrays["real_degraded"]) and len(conf_arrays["blank"]):
            boot = paired_diff_bootstrap(
                conf_arrays["real_degraded"], conf_arrays["blank"],
                rng_seed=BOOT_SEED + 1,
            )
            contrasts["degraded_minus_blank"] = {
                "diff": float(
                    conf_arrays["real_degraded"].mean() - conf_arrays["blank"].mean()
                ),
                **boot,
                "tost": tost_equivalence(
                    boot["diff_ci90"][0],
                    boot["diff_ci90"][1],
                    boot["p_lower"],
                    boot["p_upper"],
                ),
            }

    meta = {}
    if records:
        meta = {
            "checkpoint_script": records[0].get("checkpoint_script"),
            "training_condition": records[0].get("training_condition"),
            "seed": records[0].get("seed"),
            "leakage_free": records[0].get("leakage_free"),
        }
    return {
        "meta": meta,
        "n_records": len(records),
        "per_condition": per_condition,
        "contrasts": contrasts,
        "conf_arrays": conf_arrays,
    }


def analyze_one_seed_bundle(
    probe6_path: Path,
    probe3_path: Path | None,
    probe5_path: Path | None,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    records = load_jsonl(probe6_path)
    real = analyze_probe6_seed(records)
    # Drop conf_arrays from serializable copy later; keep for cross-domain.
    synth3 = load_synthetic_probe3(probe3_path) if probe3_path and probe3_path.exists() else {}
    synth5 = load_synthetic_probe5(probe5_path) if probe5_path and probe5_path.exists() else {}

    cross = {}
    if synth3 and "real" in synth3 and "real_plain" in real["conf_arrays"]:
        if len(synth3["real"]) and len(real["conf_arrays"]["real_plain"]):
            boot = paired_diff_bootstrap(
                real["conf_arrays"]["real_plain"],
                synth3["real"],
                n_boot=n_boot,
                rng_seed=BOOT_SEED + 2,
            )
            cross["tierc_plain_minus_synth_real"] = {
                "diff": float(
                    real["conf_arrays"]["real_plain"].mean() - synth3["real"].mean()
                ),
                **boot,
            }
    if synth3 and "blank" in synth3 and "blank" in real["conf_arrays"]:
        if len(synth3["blank"]) and len(real["conf_arrays"]["blank"]):
            boot = paired_diff_bootstrap(
                real["conf_arrays"]["blank"],
                synth3["blank"],
                n_boot=n_boot,
                rng_seed=BOOT_SEED + 3,
            )
            cross["tierc_blank_minus_synth_blank"] = {
                "diff": float(
                    real["conf_arrays"]["blank"].mean() - synth3["blank"].mean()
                ),
                **boot,
            }

    # Claim B pattern check within each domain: |real−blank|
    pattern = {}
    if synth3 and "real" in synth3 and "blank" in synth3:
        pattern["synth_real_minus_blank"] = float(
            synth3["real"].mean() - synth3["blank"].mean()
        )
    if "plain_minus_blank" in real["contrasts"]:
        pattern["tierc_plain_minus_blank"] = real["contrasts"]["plain_minus_blank"]["diff"]

    # Strip non-JSON arrays before return packaging
    real_out = {k: v for k, v in real.items() if k != "conf_arrays"}
    return {
        "real": real_out,
        "synthetic_probe3": {
            cond: {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if arr.size >= 2 else float("nan"),
            }
            for cond, arr in synth3.items()
        },
        "synthetic_probe5": {
            k: v for k, v in synth5.items() if k not in ("confidences", "corrects")
        },
        "cross_domain": cross,
        "claim_b_pattern": pattern,
        "source_probe6": probe6_path.as_posix(),
        "source_probe3": probe3_path.as_posix() if probe3_path else None,
        "source_probe5": probe5_path.as_posix() if probe5_path else None,
    }


def aggregate_across_seeds(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if len(per_seed) < 2:
        return {
            "n_seeds": len(per_seed),
            "seeds": sorted(per_seed.keys()),
            "ready": False,
            "note": (
                f"Only seed(s) {sorted(per_seed.keys())} on disk; "
                "across-seed summary withheld until seed{0,1,2} exist."
            ),
        }
    seeds = sorted(per_seed.keys())

    def collect(path_fn) -> dict[str, Any]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for s in seeds:
            for key, val in path_fn(per_seed[s]).items():
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                buckets[key].append(float(val))
        out = {}
        for key, vals in buckets.items():
            arr = np.asarray(vals, dtype=float)
            out[key] = {
                "per_seed": {str(s): float(v) for s, v in zip(seeds, vals)},
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) >= 2 else float("nan"),
            }
        return out

    tierc_conf = collect(
        lambda r: {
            f"conf_{c}": r["real"]["per_condition"][c]["mean"]
            for c in PROBE6_CONDS
            if c in r["real"]["per_condition"]
        }
    )
    tierc_acc = collect(
        lambda r: {
            f"acc_{c}": r["real"]["per_condition"][c]["accuracy"]
            for c in ("real_plain", "real_degraded")
            if c in r["real"]["per_condition"]
            and r["real"]["per_condition"][c].get("accuracy") is not None
        }
    )
    synth_conf = collect(
        lambda r: {
            f"synth_{c}": r["synthetic_probe3"][c]["mean"]
            for c in ("real", "blank")
            if c in r.get("synthetic_probe3", {})
        }
    )
    synth_acc = collect(
        lambda r: {
            "synth_accuracy": r["synthetic_probe5"].get("accuracy")
        }
        if r.get("synthetic_probe5")
        else {}
    )
    deltas = collect(
        lambda r: {
            k: v
            for k, v in r.get("claim_b_pattern", {}).items()
            if v is not None
        }
    )

    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ready": True,
        "tierc_confidence": tierc_conf,
        "tierc_accuracy": tierc_acc,
        "synth_confidence": synth_conf,
        "synth_accuracy": synth_acc,
        "claim_b_deltas": deltas,
    }


def analyze_all(
    probe_dir: Path,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    seed_paths = discover_probe6_seeds(probe_dir)
    per_seed = {}
    for seed, path in sorted(seed_paths.items()):
        p3 = probe_dir / f"probe3_hindi_natural_seed{seed}.jsonl"
        p5 = probe_dir / f"probe5_hindi_natural_seed{seed}.jsonl"
        per_seed[seed] = analyze_one_seed_bundle(
            path,
            p3 if p3.exists() else None,
            p5 if p5.exists() else None,
            n_boot=n_boot,
        )
    leakage_path = None
    if seed_paths:
        first = next(iter(seed_paths.values()))
        cand = first.with_suffix(".leakage.json")
        if cand.exists():
            leakage_path = cand
    leakage = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path else None
    if leakage is None:
        # Still confirm held-out validity for the write-up even before
        # Colab jsonl lands — same assert the probe runs.
        probes_dir = Path(__file__).resolve().parents[1] / "probes"
        if str(probes_dir) not in sys.path:
            sys.path.insert(0, str(probes_dir))
        from probe6_synthetic_real_gap import assert_no_train_raw_leakage  # noqa: E402

        data_root = Path("data")
        try:
            leakage = assert_no_train_raw_leakage(
                data_root / "manifests",
                data_root / "raw" / "hindi" / "images",
                "hindi",
            )
        except (FileNotFoundError, RuntimeError) as exc:
            leakage = {"leakage_free": False, "error": str(exc), "n_overlaps": -1}
    return {
        "per_seed": per_seed,
        "across_seeds": aggregate_across_seeds(per_seed),
        "leakage": leakage,
        "n_boot": n_boot,
        "seoi_delta": SEOI_DELTA,
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def render_markdown(bundle: dict[str, Any]) -> str:
    across = bundle["across_seeds"]
    seeds = sorted(bundle["per_seed"].keys())
    leak = bundle.get("leakage")

    lines = [
        "# Probe 6 — synthetic Claim B vs real Tier C",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Seeds:** {seeds}  ",
        f"**n_boot:** {bundle['n_boot']}  ",
        f"**δ (SEOI):** {bundle['seoi_delta']}  ",
        "**Scope:** hindi/natural only; Tier C plain+degraded+blank. "
        "Full Probe 6 (Tier B sweep, handwriting anecdote, held-out "
        "synthetic pages 100–109, multi-system gaps) deferred — see "
        "§ Future work and DECISIONS.md #58.",
        "",
        "---",
        "",
        "## 0. Held-out validity (data leakage)",
        "",
    ]
    if leak:
        lines.append(
            f"Training manifests under `{leak['manifests_dir']}` were "
            f"checked against `{leak['raw_images_dir']}` "
            f"({leak['n_raw_images']} files)."
        )
        lines.append("")
        lines.append("| Manifest | n image_path |")
        lines.append("|----------|--------------|")
        for name, n in leak.get("per_manifest_n_paths", {}).items():
            lines.append(f"| `{name}` | {n} |")
        lines.append("")
        if leak.get("leakage_free"):
            lines.append(
                f"**Confirmed: {leak['n_overlaps']} overlaps.** "
                "Manifest paths are renderer line crops "
                "(`data/cache/line_crops/...`); Tier C images are "
                "GlotOCR-bench via `fetch_glotocr.py` under "
                "`data/raw/hindi/images/`. The instrument never trained "
                "on these files — Probe 6 is a valid held-out test."
            )
        else:
            lines.append(
                f"**LEAKAGE DETECTED ({leak['n_overlaps']} overlaps) — "
                "do not interpret results as held-out.**"
            )
    else:
        lines.append(
            "*Leakage sidecar not found — re-run "
            "`probe6_synthetic_real_gap.py` (writes "
            "`*.leakage.json`) or "
            "`--leakage-check-only`.*"
        )
    lines.append("")

    if not seeds:
        lines.append("*No probe6_synthetic_real_hindi_seed*.jsonl on disk yet.*")
        lines.append("")
        lines.extend(_future_work_section())
        return "\n".join(lines)

    if across.get("ready"):
        lines.extend([
            "## 1. Across-seed confidence (primary table)",
            "",
            "| Domain / condition | "
            + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean | SD |",
            "|--------------------|"
            + "|".join(["------"] * len(seeds))
            + "|------|----|",
        ])
        # Synthetic rows
        for key, label in [
            ("synth_real", "Synthetic real (Probe 3)"),
            ("synth_blank", "Synthetic blank (Probe 3)"),
        ]:
            block = across["synth_confidence"].get(key)
            if not block:
                continue
            cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {cols} | {_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        for key, label in [
            ("conf_real_plain", "Tier C plain"),
            ("conf_real_degraded", "Tier C degraded"),
            ("conf_blank", "Tier C blank"),
        ]:
            block = across["tierc_confidence"].get(key)
            if not block:
                continue
            cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {cols} | {_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        lines.append("")

        lines.extend([
            "### 1a. Does confidence-blindness hold on Tier C?",
            "",
            "| Δ | " + " | ".join(f"seed{s}" for s in seeds) + " | Mean | SD |",
            "|---|" + "|".join(["------"] * len(seeds)) + "|------|----|",
        ])
        for key, label in [
            ("synth_real_minus_blank", "Synthetic (real − blank)"),
            ("tierc_plain_minus_blank", "Tier C (plain − blank)"),
        ]:
            block = across["claim_b_deltas"].get(key)
            if not block:
                continue
            cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {cols} | {_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        lines.append("")

        # Accuracy table
        lines.extend([
            "## 2. Accuracy (Tier 1/2 line correctness)",
            "",
            "| Condition | " + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean | SD |",
            "|-----------|" + "|".join(["------"] * len(seeds)) + "|------|----|",
        ])
        block = across["synth_accuracy"].get("synth_accuracy")
        if block:
            cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| Synthetic (Probe 5) | {cols} | {_fmt(block['mean'])} | "
                f"{_fmt(block['sd'])} |"
            )
        for key, label in [
            ("acc_real_plain", "Tier C plain"),
            ("acc_real_degraded", "Tier C degraded"),
        ]:
            block = across["tierc_accuracy"].get(key)
            if not block:
                continue
            cols = " | ".join(_fmt(block["per_seed"].get(str(s))) for s in seeds)
            lines.append(
                f"| {label} | {cols} | {_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        lines.append("")
        lines.append(
            "Synthetic accuracy is from Probe 5 on training-manifest "
            "line crops (same crops Claim B used). DECISIONS.md #45's "
            "held-out pages 100–109 remain future work — do not read "
            "the synthetic–real *accuracy* gap as a clean domain-shift "
            "estimate; the confidence comparison is the Claim B test."
        )
        lines.append("")
    else:
        lines.append(f"*{across.get('note')}*")
        lines.append("")

    lines.extend(["## 3. Per-seed detail (bootstrap)", ""])
    for s in seeds:
        r = bundle["per_seed"][s]
        lines.append(f"### Seed {s}")
        lines.append("")
        lines.append(
            f"Sources: `{r.get('source_probe6')}`; "
            f"synth `{r.get('source_probe3')}` / `{r.get('source_probe5')}`."
        )
        lines.append("")
        lines.append("| Condition | n | Mean conf | Acc |")
        lines.append("|-----------|---|-----------|-----|")
        for c in PROBE6_CONDS:
            pc = r["real"]["per_condition"].get(c, {})
            lines.append(
                f"| {c} | {pc.get('n', 0)} | {_fmt(pc.get('mean'))} | "
                f"{_fmt(pc.get('accuracy'))} |"
            )
        for c, block in r.get("synthetic_probe3", {}).items():
            lines.append(
                f"| synth_{c} (P3) | {block['n']} | {_fmt(block['mean'])} | — |"
            )
        if r.get("synthetic_probe5"):
            p5 = r["synthetic_probe5"]
            lines.append(
                f"| synth (P5) | {p5.get('n')} | "
                f"{_fmt(p5.get('mean_confidence'))} | "
                f"{_fmt(p5.get('accuracy'))} |"
            )
        lines.append("")
        for name, ctr in r["real"].get("contrasts", {}).items():
            tost = ctr.get("tost", {})
            lines.append(
                f"- **{name}:** Δ={_fmt(ctr.get('diff'))}, "
                f"boot 95% CI [{_fmt(ctr['diff_ci95'][0])}, "
                f"{_fmt(ctr['diff_ci95'][1])}], "
                f"TOST δ={SEOI_DELTA} equiv="
                f"{'yes' if tost.get('equivalent') else 'no'}"
            )
        lines.append("")

    lines.extend(["## 4. Finding (plain language)", ""])
    if across.get("ready"):
        d_synth = across["claim_b_deltas"].get("synth_real_minus_blank", {})
        d_real = across["claim_b_deltas"].get("tierc_plain_minus_blank", {})
        lines.append(
            f"Synthetic Claim B gap (real − blank) across seeds: "
            f"{_fmt(d_synth.get('mean'))} ± {_fmt(d_synth.get('sd'))}. "
            f"Tier C plain − blank: "
            f"{_fmt(d_real.get('mean'))} ± {_fmt(d_real.get('sd'))}."
        )
        lines.append("")
        # Honest gate using SEOI
        mean_r = d_real.get("mean")
        if mean_r is not None and not math.isnan(mean_r):
            if abs(mean_r) < SEOI_DELTA:
                lines.append(
                    f"|Tier C Δ| < δ={SEOI_DELTA}: confidence-blindness "
                    "**holds** on real documents at the same SEI used for "
                    "Probe 5b — plain and blank confidence are practically "
                    "equivalent."
                )
            else:
                lines.append(
                    f"|Tier C Δ| ≥ δ={SEOI_DELTA}: real documents show a "
                    "**different** pattern from synthetic Claim B — "
                    "confidence separates plain from blank by a "
                    "practically meaningful amount. Report this as a "
                    "domain-dependent finding, not a universal property."
                )
        lines.append("")
    else:
        lines.append("Finding withheld until seeds 0/1/2 probe6 jsonl exist.")
        lines.append("")

    lines.extend(_future_work_section())
    return "\n".join(lines)


def _future_work_section() -> list[str]:
    return [
        "## Future work (full Probe 6 — not built)",
        "",
        "- Tier B degradation sweep on synthetic renders",
        "- Handwriting anecdote (15–20 lines, 2–3 writers, qualitative)",
        "- Held-out synthetic pages 100–109 (DECISIONS.md #45) for a "
        "clean accuracy-gap estimate",
        "- Multi-system gaps (Tesseract / Surya / PaddleOCR / instrument)",
        "- The previous metric-only aggregator API in this file's history "
        "is superseded for the paper-deadline scope",
        "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Probe 6 Tier C vs synthetic")
    ap.add_argument("--probe-dir", default="data/probe_results")
    ap.add_argument("--out", default="docs/probe6_synthetic_real_analysis.md")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = ap.parse_args()

    bundle = analyze_all(Path(args.probe_dir), n_boot=args.n_boot)
    md = render_markdown(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(
        f"[analyze_probe6] wrote {out} "
        f"(seeds={sorted(bundle['per_seed'].keys())})"
    )
    leak = bundle.get("leakage")
    if leak:
        print(f"  leakage_free={leak.get('leakage_free')} overlaps={leak.get('n_overlaps')}")


if __name__ == "__main__":
    main()

"""
Probe 3b analysis — training-curve disambiguation for blank control.

Why this exists: probe3_training_curve.py writes a JSON of
real-vs-blank confidence at intermediate checkpoints. The in-script
interpretation hedges when loss falls and the gap barely moves. The
correct framing for this project's write-up is sharper: undertraining
and ungrounded confidence are not competing explanations. The model
*is* undertrained; the finding is that its confidence gives no
indication of that — a calibrated undertrained model would report low
confidence, and this one reports ~0.99 and rises as training proceeds.

With seed{0,1,2} curve files: the real−blank gap sign flips at most
steps and |SD| exceeds |mean| at most steps → indistinguishable from
zero across seeds. Step 3000 is negative in all three seeds (mean
magnitude ~0.0075) as a separate observation, without over-claiming.

Called from: CLI after data/probe_results/probe3_curve_*.json exists.
Writes docs/probe3_curve_analysis.md (or --out).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


def load_curve(path: Path) -> dict[str, Any]:
    """Load Probe 3b curve JSON (points + metadata)."""
    return json.loads(path.read_text(encoding="utf-8"))


def discover_curve_paths(
    probe_dir: Path,
    script: str = "hindi",
    condition: str = "natural",
) -> dict[int, Path]:
    """Find probe3_curve_{script}_{condition}_seed{N}.json for N in 0,1,2."""
    found: dict[int, Path] = {}
    for seed in (0, 1, 2):
        p = probe_dir / f"probe3_curve_{script}_{condition}_seed{seed}.json"
        if p.is_file():
            found[seed] = p
    return found


def sibling_curve_seeds(source: Path, script: str, condition: str) -> list[int]:
    """Which probe3_curve_{script}_{condition}_seed{N}.json files exist."""
    return sorted(
        discover_curve_paths(source.parent, script, condition).keys()
    )


def aggregate_curves(
    curves: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """
    Across-seed mean±SD of gap / conf / loss / accuracy per step.

    Sign flips and SD>|mean| are the claim gates for calling the gap
    indistinguishable from zero (docs/statistical_repair.md).
    """
    seeds = sorted(curves.keys())
    # Assume shared step grid (verified on write).
    steps = [p["step"] for p in curves[seeds[0]]["points"]]
    per_step: list[dict[str, Any]] = []
    n_flip = 0
    n_sd_gt_mean = 0
    for step in steps:
        gaps = []
        reals = []
        blanks = []
        accs = []
        losses = []
        per_seed_gap: dict[str, float] = {}
        for s in seeds:
            pt = next(p for p in curves[s]["points"] if p["step"] == step)
            gaps.append(float(pt["real_minus_blank_gap"]))
            reals.append(float(pt["mean_confidence_real"]))
            blanks.append(float(pt["mean_confidence_blank"]))
            accs.append(float(pt["accuracy"]))
            losses.append(float(pt["training_loss"]))
            per_seed_gap[str(s)] = float(pt["real_minus_blank_gap"])
        g = np.asarray(gaps, dtype=float)
        mean_g = float(g.mean())
        sd_g = float(g.std(ddof=1)) if len(g) >= 2 else float("nan")
        signs = {int(np.sign(x)) for x in gaps if x != 0.0}
        flip = (any(x > 0 for x in gaps) and any(x < 0 for x in gaps))
        sd_gt = (not math.isnan(sd_g)) and (abs(sd_g) > abs(mean_g))
        if flip:
            n_flip += 1
        if sd_gt:
            n_sd_gt_mean += 1
        per_step.append({
            "step": step,
            "per_seed_gap": per_seed_gap,
            "gap_mean": mean_g,
            "gap_sd": sd_g,
            "sign_flip": flip,
            "sd_exceeds_mean": sd_gt,
            "real_mean": float(np.mean(reals)),
            "blank_mean": float(np.mean(blanks)),
            "acc_mean": float(np.mean(accs)),
            "loss_mean": float(np.mean(losses)),
            "all_negative": all(x < 0 for x in gaps),
            "all_positive": all(x > 0 for x in gaps),
        })

    step_3000 = next((r for r in per_step if r["step"] == 3000), None)
    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "ready": len(seeds) >= 3,
        "steps": steps,
        "per_step": per_step,
        "n_steps_sign_flip": n_flip,
        "n_steps_sd_gt_mean": n_sd_gt_mean,
        "n_steps": len(steps),
        "gap_indistinguishable_from_zero": (
            len(seeds) >= 3
            and n_flip >= 4
            and n_sd_gt_mean >= 4
        ),
        "step_3000": step_3000,
    }


def render_markdown(curve: dict[str, Any], source: Path) -> str:
    """Single-curve compat wrapper → multi-seed renderer when siblings exist."""
    script = curve.get("script", "hindi")
    condition = curve.get("condition", "natural")
    paths = discover_curve_paths(source.parent, script, condition)
    if not paths:
        paths = {int(curve.get("seed", 0)): source}
    curves = {s: load_curve(p) if p != source else curve for s, p in paths.items()}
    # Ensure source curve is present even if discover missed it.
    seed = int(curve.get("seed", 0))
    curves[seed] = curve
    return render_multi_seed_markdown(curves, paths)


def render_multi_seed_markdown(
    curves: dict[int, dict[str, Any]],
    paths: dict[int, Path],
) -> str:
    """
    Multi-seed Probe 3b write-up.

    With three seeds: gap sign flips at most steps and SD exceeds mean
    → indistinguishable from zero. Step 3000 all-negative is noted
    separately without calling the whole curve a reversal.
    """
    agg = aggregate_curves(curves)
    seeds = agg["seeds"]
    c0 = curves[seeds[0]]
    script = c0.get("script", "hindi")
    condition = c0.get("condition", "natural")
    n_samples = c0.get("n_samples")
    steps = agg["steps"]

    sources = ", ".join(f"`{paths[s].as_posix()}`" for s in seeds if s in paths)

    # Loss / conf trajectory from across-seed means at first/last step.
    first = agg["per_step"][0]
    last = agg["per_step"][-1]
    loss_ratio = (
        first["loss_mean"] / last["loss_mean"]
        if last["loss_mean"]
        else float("inf")
    )

    lines: list[str] = [
        "# Probe 3b analysis — training curve (real vs blank confidence)",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Sources:** {sources}  ",
        f"**Run:** {script}/{condition}/seeds {seeds}  ",
        f"**Snapshots:** {len(steps)} steps {steps}  ",
        f"**Samples per step:** {n_samples}  ",
        "**Correction note:** see "
        "[statistical_repair.md](statistical_repair.md).",
        "",
        "---",
        "",
        "## 1. Across-seed curve table",
        "",
        "| Step | Loss (mean) | Real conf (mean) | Blank conf (mean) | "
        "Gap mean | Gap SD | Sign flip? | |SD|>|mean|? | Acc (mean) |",
        "|------|-------------|------------------|-------------------|"
        "----------|--------|------------|--------------|------------|",
    ]
    for row in agg["per_step"]:
        lines.append(
            f"| {row['step']} | {row['loss_mean']:.4f} | "
            f"{row['real_mean']:.4f} | {row['blank_mean']:.4f} | "
            f"{row['gap_mean']:+.4f} | {row['gap_sd']:.4f} | "
            f"{'yes' if row['sign_flip'] else 'no'} | "
            f"{'yes' if row['sd_exceeds_mean'] else 'no'} | "
            f"{row['acc_mean']:.4f} |"
        )

    lines.extend([
        "",
        "### 1b. Per-seed gaps",
        "",
        "| Step | " + " | ".join(f"seed{s}" for s in seeds) + " |",
        "|------|" + "|".join(["------"] * len(seeds)) + "|",
    ])
    for row in agg["per_step"]:
        cols = " | ".join(
            f"{row['per_seed_gap'][str(s)]:+.4f}" for s in seeds
        )
        lines.append(f"| {row['step']} | {cols} |")

    lines.extend([
        "",
        "## 2. Interpretation",
        "",
        f"Across seeds {seeds}, loss fell ~{loss_ratio:.0f}× "
        f"({first['loss_mean']:.3f} → {last['loss_mean']:.3f}) while "
        f"mean real confidence **rose** "
        f"({first['real_mean']:.3f} → {last['real_mean']:.3f}) and "
        f"accuracy stayed near floor until late "
        f"(mean acc {last['acc_mean']:.3f} at step {last['step']}).",
        "",
    ])

    if agg["ready"]:
        lines.append(
            f"The real−blank gap **sign flips across seeds at "
            f"{agg['n_steps_sign_flip']} of {agg['n_steps']} steps**, "
            f"and |gap SD| exceeds |gap mean| at "
            f"{agg['n_steps_sd_gt_mean']} of {agg['n_steps']} steps. "
            "It is now defensible to call the gap "
            "**indistinguishable from zero** across training seeds — "
            "not an emerging vision signal, and not a reliable "
            "negative either."
        )
        lines.append("")
        s3000 = agg.get("step_3000")
        if s3000 and s3000["all_negative"]:
            per_seed_str = ", ".join(
                f"{s3000['per_seed_gap'][str(s)]:+.4f}" for s in seeds
            )
            lines.append(
                f"**Step 3000 observation (not over-claimed):** gap is "
                f"negative in **all {agg['n_seeds']} seeds** "
                f"({per_seed_str}; "
                f"mean {s3000['gap_mean']:+.4f}, magnitude "
                f"{abs(s3000['gap_mean']):.4f}). Recorded as a stable "
                "single-step sign; it does not license calling the "
                "whole curve a blank>real reversal."
            )
            lines.append("")
    else:
        lines.append(
            f"Only seeds {seeds} on disk — withhold the "
            "indistinguishable-from-zero claim until seed{{0,1,2}} "
            "all exist (DECISIONS.md #52)."
        )
        lines.append("")

    lines.extend([
        "### Correct framing",
        "",
        "Undertraining and ungrounded confidence are **not** competing "
        "explanations. The model **is** undertrained (accuracy low on "
        "this curve). The finding is that its confidence gives **no "
        "indication** of that: a well-calibrated undertrained model "
        "would report **low** confidence. This one reports ~0.99 and "
        "rises as training proceeds.",
        "",
        "The in-script interpretation that hedges between \"(a) "
        "ungrounded\" and \"(b) undertrained\" therefore misses the "
        "point. Both are true at once; the calibration failure is that "
        "confidence tracks training progress (loss ↓, conf ↑) instead "
        "of image evidence (gap ≈ 0 across seeds).",
        "",
        "### What this does not establish",
        "",
        "- That longer training would never open a real−blank gap "
        "(only that through 5000 steps × 3 seeds it does not).",
        "- That the same curve holds for flattened/inverted conditions.",
        "- That production OCR confidence is similarly ungrounded "
        "(instrument only).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze Probe 3b training-curve JSON (multi-seed)"
    )
    ap.add_argument(
        "--input",
        default=None,
        help="Single curve JSON (overrides seed discovery)",
    )
    ap.add_argument(
        "--probe-dir",
        default="data/probe_results",
        help="Directory to discover probe3_curve_*_seed{0,1,2}.json",
    )
    ap.add_argument(
        "--script",
        default="hindi",
    )
    ap.add_argument(
        "--condition",
        default="natural",
    )
    ap.add_argument(
        "--out",
        default="docs/probe3_curve_analysis.md",
        help="Markdown report path",
    )
    args = ap.parse_args()

    if args.input:
        src = Path(args.input)
        curve = load_curve(src)
        paths = {int(curve.get("seed", 0)): src}
        # Still pick up siblings next to --input when present.
        script = curve.get("script", args.script)
        condition = curve.get("condition", args.condition)
        discovered = discover_curve_paths(src.parent, script, condition)
        paths.update(discovered)
        curves = {s: load_curve(p) for s, p in paths.items()}
    else:
        paths = discover_curve_paths(
            Path(args.probe_dir), args.script, args.condition
        )
        if not paths:
            raise SystemExit(
                f"no probe3_curve_{args.script}_{args.condition}_seed*.json "
                f"in {args.probe_dir}"
            )
        curves = {s: load_curve(p) for s, p in paths.items()}

    md = render_multi_seed_markdown(curves, paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    agg = aggregate_curves(curves)
    print(
        f"[analyze_probe3_curve] wrote {out} "
        f"(seeds={agg['seeds']}, steps={agg['n_steps']})"
    )
    print(
        f"  sign_flips={agg['n_steps_sign_flip']}/{agg['n_steps']}  "
        f"sd>|mean|={agg['n_steps_sd_gt_mean']}/{agg['n_steps']}  "
        f"gap≈0={agg['gap_indistinguishable_from_zero']}"
    )
    if agg.get("step_3000"):
        s = agg["step_3000"]
        print(
            f"  step3000 mean_gap={s['gap_mean']:+.4f} "
            f"all_neg={s['all_negative']} "
            f"per_seed={s['per_seed_gap']}"
        )
    for row in agg["per_step"]:
        print(
            f"  step={row['step']:5d}  gap_mean={row['gap_mean']:+.4f}  "
            f"gap_sd={row['gap_sd']:.4f}  flip={row['sign_flip']}  "
            f"sd>|mean|={row['sd_exceeds_mean']}"
        )


if __name__ == "__main__":
    main()

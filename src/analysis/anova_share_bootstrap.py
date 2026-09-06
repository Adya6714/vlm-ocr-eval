"""
Seed-respecting bootstrap CIs on ANOVA variance shares (Step 6).

Why this exists: Follow-Up 5 in docs/paper_defensibility_stats.md reports
point shares (image / seed / residual) with no intervals. This script
resamples images with replacement (same indices across the three seeds
so the 60×3 panel stays aligned), recomputes anova_var_decomp, 2,000
replicates, empirical 2.5/97.5 percentiles.

That is the same image-within-panel resampling used for seed-clustered
CIs on headline means in Offline Analysis 2 of that file, applied to
the three variance components.

Output: docs/variance_share_bootstrap.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from paper_defensibility_stats import anova_var_decomp, panel_from_jsonl  # noqa: E402

N_BOOT = 2000
SEED = 0


def bootstrap_shares(Y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """
    Y: (n_images, n_seeds). Resample image rows with replacement.
    """
    rng = np.random.default_rng(seed)
    n_i = Y.shape[0]
    pcts = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n_i, size=n_i)
        d = anova_var_decomp(Y[idx])
        pcts[b] = [d["pct_image"], d["pct_seed"], d["pct_resid"]]
    lo = np.percentile(pcts, 2.5, axis=0)
    hi = np.percentile(pcts, 97.5, axis=0)
    return {"lo": lo, "hi": hi}


def main() -> None:
    repo = _ROOT
    specs = [
        (
            "Probe 5b Hindi Confidence",
            "data/probe_results/probe5b_hindi_natural_seed{}.jsonl",
            "hindi",
            "mean_confidence",
        ),
        (
            "Probe 5b Blank Confidence",
            "data/probe_results/probe5b_hindi_natural_seed{}.jsonl",
            "blank",
            "mean_confidence",
        ),
        (
            "GT Likelihood (real)",
            "data/probe_results/probe_gt_likelihood_hindi_natural_seed{}.jsonl",
            "real",
            "mean_log_p_gt",
        ),
        (
            "GT Likelihood (blank)",
            "data/probe_results/probe_gt_likelihood_hindi_natural_seed{}.jsonl",
            "blank",
            "mean_log_p_gt",
        ),
    ]
    lines = [
        "# Bootstrap CIs on ANOVA variance shares (Step 6)",
        "",
        f"**Method:** {N_BOOT} replicates; resample 60 image rows with replacement;",
        "keep all three seeds for each drawn row; `anova_var_decomp` as in",
        "`src/analysis/paper_defensibility_stats.py`. RNG seed 0.",
        "Point shares are the original Follow-Up 5 numbers (same function, full panel).",
        "",
        "| Panel | Image % [95% CI] | Seed % [95% CI] | Residual % [95% CI] |",
        "|---|---|---|---|",
    ]
    for label, pat, cond, key in specs:
        Y = panel_from_jsonl(str(repo / pat), cond, key)
        d = anova_var_decomp(Y)
        b = bootstrap_shares(Y)
        lines.append(
            f"| {label} | {d['pct_image']:.1f} [{b['lo'][0]:.1f}, {b['hi'][0]:.1f}] | "
            f"{d['pct_seed']:.1f} [{b['lo'][1]:.1f}, {b['hi'][1]:.1f}] | "
            f"{d['pct_resid']:.1f} [{b['lo'][2]:.1f}, {b['hi'][2]:.1f}] |"
        )
    out = repo / "docs" / "variance_share_bootstrap.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

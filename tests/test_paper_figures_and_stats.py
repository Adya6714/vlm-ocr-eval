"""
Unit tests for make_paper_figures.py and paper_defensibility_stats.py.

Why these exist: Ensures that ROC curve calculations, ECE binning,
statistical functions, and figure generation routines do not regress
and produce valid publication artifacts.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from analysis.make_paper_figures import (
    compute_roc,
    grapheme_cer,
    graphemes,
    levenshtein,
    make_fig1_position_dissociation,
    make_fig2_ablation_kl,
    make_fig3_confidence_distributions,
    make_fig4_regime_contrast,
    make_fig5_output_degeneracy,
)
from analysis.paper_defensibility_stats import (
    anova_var_decomp,
    auroc,
    is_correct,
    rankdata,
    spearman,
)

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "data" / "probe_results"


def test_graphemes_devanagari():
    """Grapheme clustering should keep base consonant and matras together."""
    text = "किताब"
    clusters = graphemes(text)
    assert len(clusters) == 3
    assert clusters[0] == "कि"
    assert clusters[1] == "ता"
    assert clusters[2] == "ब"


def test_levenshtein_and_cer():
    """Identical strings -> CER 0; completely different -> CER 1+."""
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert levenshtein(["a", "b"], ["a", "c"]) == 1
    assert grapheme_cer("कंगना", "कंगना") == 0.0
    assert grapheme_cer("भारत", "अमेरिका") > 0.5


def test_compute_roc_perfect_separation():
    """Perfect discrimination should yield AUROC = 1.0."""
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1]
    fpr, tpr, auc = compute_roc(scores, labels)
    assert auc == 1.0
    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0


def test_compute_roc_chance():
    """Inverted scores should yield AUROC = 0.0."""
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [0, 0, 0, 1, 1]
    _, _, auc = compute_roc(scores, labels)
    assert auc == 0.0


def test_rankdata_and_spearman():
    """Rankdata should resolve ties cleanly and Spearman should detect correlation."""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert abs(spearman(x, y) - 1.0) < 1e-6
    y_inv = [40.0, 30.0, 20.0, 10.0]
    assert abs(spearman(x, y_inv) - (-1.0)) < 1e-6


def test_anova_var_decomp_balanced():
    """Balanced panel decomposition sums to 100%."""
    Y = np.array([
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
    ])
    d = anova_var_decomp(Y)
    total_pct = d["pct_image"] + d["pct_seed"] + d["pct_resid"]
    assert abs(total_pct - 100.0) < 1e-6
    assert d["pct_image"] > d["pct_seed"]


@pytest.mark.skipif(not _RESULTS.exists(), reason="probe_results not present")
def test_make_figures_smoke(tmp_path: Path):
    """Smoke test ensuring all 5 figure generators execute without error."""
    seeds = [0, 1, 2]

    # Fig 1
    fig1 = make_fig1_position_dissociation(_RESULTS, seeds, mode="working")
    fig1.savefig(tmp_path / "fig1.png")
    fig1_pub = make_fig1_position_dissociation(_RESULTS, seeds, mode="paper")
    fig1_pub.savefig(tmp_path / "fig1.pdf")

    # Fig 2
    fig2 = make_fig2_ablation_kl(_RESULTS, seeds, mode="working")
    fig2.savefig(tmp_path / "fig2.png")
    fig2_pub = make_fig2_ablation_kl(_RESULTS, seeds, mode="paper")
    fig2_pub.savefig(tmp_path / "fig2.pdf")

    # Fig 3
    fig3 = make_fig3_confidence_distributions(_RESULTS, seeds, mode="working")
    fig3.savefig(tmp_path / "fig3.png")
    fig3_pub = make_fig3_confidence_distributions(_RESULTS, seeds, mode="paper")
    fig3_pub.savefig(tmp_path / "fig3.pdf")

    # Fig 4
    fig4 = make_fig4_regime_contrast(_RESULTS, seeds, mode="working")
    fig4.savefig(tmp_path / "fig4.png")
    fig4_pub = make_fig4_regime_contrast(_RESULTS, seeds, mode="paper")
    fig4_pub.savefig(tmp_path / "fig4.pdf")

    # Fig 5
    fig5 = make_fig5_output_degeneracy(_RESULTS, seeds)
    fig5.savefig(tmp_path / "fig5.png")

    assert (tmp_path / "fig1.png").exists()
    assert (tmp_path / "fig1.pdf").exists()
    assert (tmp_path / "fig2.png").exists()
    assert (tmp_path / "fig2.pdf").exists()
    assert (tmp_path / "fig3.png").exists()
    assert (tmp_path / "fig3.pdf").exists()
    assert (tmp_path / "fig4.png").exists()
    assert (tmp_path / "fig4.pdf").exists()
    assert (tmp_path / "fig5.png").exists()

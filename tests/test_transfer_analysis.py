"""
Checkable properties of Stage 5b rank correlation and the spend gate.

Why synthetic vectors only: Decision #89 forbids peeking at the live
coefficient before the user confirms the page set. Tests must not load
sarvam_transfer_probe.jsonl and compute ρ.
"""

from __future__ import annotations

import numpy as np

from eval.transfer_analysis import spearman_perm_p
from probes import stage5b_sarvam_pages as s5b


def test_spearman_perm_perfect_positive_is_extreme():
    """Monotone pairs → ρ=1; almost every shuffle is less extreme."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    r = spearman_perm_p(x, y, n_perm=500, seed=0)
    assert abs(r["rho"] - 1.0) < 1e-9
    assert r["n"] == 8
    assert r["p_value"] < 0.02


def test_spearman_perm_uncorrelated_not_tiny_p():
    """Independent Gaussian ranks should not look like a locked transfer."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=30).tolist()
    y = rng.normal(size=30).tolist()
    r = spearman_perm_p(x, y, n_perm=400, seed=0)
    assert 0.0 < r["p_value"] <= 1.0
    assert abs(r["rho"]) < 0.6


def test_spend_gate_rejects_wrong_rupees(tmp_path, monkeypatch):
    """Wrong --i-confirm-spend-inr must not instantiate a client / POST."""
    repo = tmp_path / "repo"
    (repo / "data/probe_results").mkdir(parents=True)
    (repo / "data/cache/sarvam").mkdir(parents=True)
    (repo / "data/raw/hindi/images").mkdir(parents=True)

    monkeypatch.setattr(s5b, "hindi_probe5b_ids", lambda _root: ["1", "2"])
    monkeypatch.setattr(s5b, "stage5a_hindi_plain_ids", lambda _root: set())

    def fake_gt(_root):
        return {
            "1": {
                "text": "a",
                "img_plain_path": "data/raw/hindi/images/1_plain.png",
                "img_degraded_path": "data/raw/hindi/images/1_degraded.png",
            },
            "2": {
                "text": "b",
                "img_plain_path": "data/raw/hindi/images/2_plain.png",
                "img_degraded_path": "data/raw/hindi/images/2_degraded.png",
            },
        }

    monkeypatch.setattr(s5b, "load_hindi_gt", fake_gt)
    for iid in ("1", "2"):
        (repo / f"data/raw/hindi/images/{iid}_plain.png").write_bytes(
            b"plain" + iid.encode()
        )
        (repo / f"data/raw/hindi/images/{iid}_degraded.png").write_bytes(
            b"deg" + iid.encode()
        )

    called = {"client": False}

    def boom(*_a, **_k):
        called["client"] = True
        raise AssertionError("SarvamClient must not be constructed")

    monkeypatch.setattr("eval.sarvam_client.SarvamClient", boom)

    rc = s5b.run_fetch(
        repo,
        "hindi-plain-remaining",
        repo / "data/probe_results/sarvam_stage5b_pages.jsonl",
        confirm_inr=0.01,
    )
    assert rc == 2
    assert called["client"] is False


def test_dry_run_zero_without_confirm(tmp_path, monkeypatch):
    """No confirm flag → exit 0 and no jsonl writes."""
    repo = tmp_path / "repo"
    (repo / "data/probe_results").mkdir(parents=True)
    (repo / "data/cache/sarvam").mkdir(parents=True)
    (repo / "data/raw/hindi/images").mkdir(parents=True)
    monkeypatch.setattr(s5b, "hindi_probe5b_ids", lambda _root: ["1"])
    monkeypatch.setattr(s5b, "stage5a_hindi_plain_ids", lambda _root: set())
    monkeypatch.setattr(
        s5b,
        "load_hindi_gt",
        lambda _root: {
            "1": {
                "text": "a",
                "img_plain_path": "data/raw/hindi/images/1_plain.png",
                "img_degraded_path": "data/raw/hindi/images/1_degraded.png",
            }
        },
    )
    (repo / "data/raw/hindi/images/1_plain.png").write_bytes(b"x")
    (repo / "data/raw/hindi/images/1_degraded.png").write_bytes(b"y")
    out = repo / "data/probe_results/sarvam_stage5b_pages.jsonl"
    rc = s5b.run_fetch(repo, "hindi-plain-remaining", out, confirm_inr=None)
    assert rc == 0
    assert not out.exists()

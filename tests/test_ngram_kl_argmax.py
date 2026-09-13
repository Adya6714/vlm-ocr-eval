"""Tests for 5-gram → tokenizer scatter and Step 4b helpers (no checkpoints)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from analysis.ngram_kl_argmax import (
    EOS,
    context_at_step,
    kl_np,
    ngram_next_dist,
    scatter_onto_tokenizer,
)
from analysis.position_matched_ngrams import BOS, graphemes


def test_context_at_step_matches_teacher_forced_content_then_eos():
    # "कि" is one grapheme cluster in Devanagari.
    text = "कब"
    gs = graphemes(text)
    assert len(gs) == 2
    ctx0 = context_at_step(text, 0, n=5)
    assert ctx0 == (BOS, BOS, BOS, BOS)
    ctx1 = context_at_step(text, 1, n=5)
    assert ctx1 == (BOS, BOS, BOS, gs[0])
    ctx_eos = context_at_step(text, 2, n=5)
    assert ctx_eos == (BOS, BOS, gs[0], gs[1])


def test_scatter_unknown_grapheme_goes_to_rare():
    dist = {"क": 0.7, " unseen ": 0.3}
    cluster_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<RARE>": 3, "क": 4}
    q = scatter_onto_tokenizer(dist, cluster_to_id, vocab_size=5, rare_id=3, eps=1e-30)
    # After floor+renorm the bulk should sit on 4 and 3.
    assert q.argmax() == 4
    assert q[3] > q[0]


def test_identical_distributions_kl_near_zero():
    p = np.array([0.1, 0.2, 0.7])
    assert kl_np(p, p) < 1e-8


def test_ngram_next_dist_add_alpha():
    counts = Counter({(BOS, BOS, BOS, BOS, "क"): 3})
    context_counts = Counter({(BOS, BOS, BOS, BOS): 3})
    alphabet = {"क", EOS}
    v_est = 4
    dist = ngram_next_dist((BOS, BOS, BOS, BOS), counts, context_counts, alphabet, v_est, alpha=0.01)
    assert dist["क"] > dist[EOS]
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9) or sum(dist.values()) < 1.0 + 1e-6

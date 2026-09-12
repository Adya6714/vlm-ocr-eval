"""
RLVR reward for the demo: verifiable terms minus a coverage penalty.

Why this exists: SFT teaches token-level imitation. RLVR scores a
*whole* reading (character accuracy + structure + reading-order tau)
so the model can be pushed toward document-level quality. Decision #11
scopes ablations to **one** experiment: drop the coverage term and
check whether the policy omits text to game accuracy.

Why coverage is subtractive: if reward is only accuracy on emitted
text, omitting rare glyphs raises the score. Coverage = fraction of
GT grapheme clusters that appear in the hypothesis (order-insensitive
multiset overlap). Reward:

    R = char_acc + teds + tau - lambda_coverage * (1 - coverage)

lambda_coverage=0 is the ablation. This file computes R; it does not
run PPO/GRPO. Training is blocked on a finished SFT adapter (Step 2).

TEDS: if no table tree is supplied, teds=0 so the term is inert rather
than invented. Tau: if no block orders, tau=0.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import regex


def grapheme_clusters(text: str) -> list[str]:
    return regex.findall(r"\X", text or "")


def char_accuracy(hyp: str, gt: str) -> float:
    """
    1 - grapheme Levenshtein / max(len_gt, 1), clipped to [0, 1].

    Length-normalized against GT so a short hyp cannot get a free 1.0
    merely by matching a prefix *if we also use coverage* — without
    coverage, a hyp that is an exact prefix still scores well on this
    term when GT is longer (deletions count). Empty hyp vs nonempty GT
    is 0.0 here; the omission game shows up as high precision-style
    scores only if a caller used a different accuracy. We still expose
    `emitted_only_accuracy` for the ablation diagnostic.
    """
    h = grapheme_clusters(hyp)
    g = grapheme_clusters(gt)
    if not g and not h:
        return 1.0
    if not g:
        return 0.0
    dist = _levenshtein(h, g)
    return max(0.0, 1.0 - dist / max(len(g), 1))


def emitted_only_accuracy(hyp: str, gt: str) -> float:
    """
    Accuracy that only looks at what was emitted (prefix-friendly).

    This is the gaming channel: hyp='easy' against a long GT that
    starts with 'easy' scores 1.0 here and ~0 coverage.
    """
    h = grapheme_clusters(hyp)
    g = grapheme_clusters(gt)
    if not h:
        return 1.0 if not g else 0.0
    # longest prefix of hyp that matches GT prefix / alignment: use
    # hyp as reference so extra GT is not penalized.
    dist = _levenshtein(h, g[: len(h)] if len(g) >= len(h) else g)
    return max(0.0, 1.0 - dist / max(len(h), 1))


def coverage(hyp: str, gt: str) -> float:
    """Multiset overlap of GT graphemes found in hyp, over |GT|."""
    g = grapheme_clusters(gt)
    if not g:
        return 1.0
    hc = Counter(grapheme_clusters(hyp))
    used = 0
    for ch in g:
        if hc[ch] > 0:
            hc[ch] -= 1
            used += 1
    return used / len(g)


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def compute_reward(
    hyp: str,
    gt: str,
    *,
    tau: float = 0.0,
    teds: float = 0.0,
    lambda_coverage: float = 1.0,
    use_emitted_only_acc: bool = False,
) -> dict:
    """
    Scalar reward plus terms. lambda_coverage=0 is Decision #11 ablation.
    """
    acc = (
        emitted_only_accuracy(hyp, gt)
        if use_emitted_only_acc
        else char_accuracy(hyp, gt)
    )
    cov = coverage(hyp, gt)
    reward = acc + teds + tau - lambda_coverage * (1.0 - cov)
    return {
        "reward": reward,
        "char_acc": acc,
        "teds": teds,
        "tau": tau,
        "coverage": cov,
        "lambda_coverage": lambda_coverage,
    }


def ablation_omission_signal(gt: str, easy_prefix: str) -> dict:
    """
    Checkable claim: with lambda=0 and emitted-only accuracy, omitting
    the hard suffix can outscore a full (slightly noisy) reading.
    """
    omit = compute_reward(
        easy_prefix, gt, lambda_coverage=0.0, use_emitted_only_acc=True
    )
    # A complete reading that slightly mangles the hard tail: this is
    # the policy the coverage term is supposed to protect. Without
    # coverage, a perfect easy prefix can beat it.
    noisy_full = easy_prefix + " " + "x" * max(1, len(gt) - len(easy_prefix))
    if noisy_full == gt:
        noisy_full = gt[:-1] + ("?" if not gt.endswith("?") else "!")
    full = compute_reward(
        noisy_full, gt, lambda_coverage=0.0, use_emitted_only_acc=True
    )
    omit_cov = compute_reward(
        easy_prefix, gt, lambda_coverage=1.0, use_emitted_only_acc=True
    )
    return {
        "omit": omit,
        "full": full,
        "omit_with_coverage": omit_cov,
        "omit_beats_full_without_coverage": omit["reward"] > full["reward"] - 1e-9
        and omit["coverage"] < full["coverage"],
        "coverage_term_penalizes_omit": omit_cov["reward"] < omit["reward"],
    }

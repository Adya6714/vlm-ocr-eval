"""
Glyph-cluster frequency control — the Stage 1 knob Probe 1 turns.

Why this exists: Probe 1 asks whether accuracy tracks *exposure* or
*visual complexity*. That question is only answerable if exposure is a
dial we can set, not a property of whatever corpus we happened to
download. This module is that dial: given a source corpus and a mode
(`natural` / `flattened` / `inverted`), it emits a new corpus whose
realized grapheme-cluster histogram matches the target within a
declared TV tolerance (see IMPLEMENTATION.md Stage 1 acceptance).

Where it sits: third Stage 1 module. Upstream is any list of source
strings (GlotOCR ground truth, Wikipedia extracts, …). Downstream,
`render.py` paints the resampled text into a layout template. The
instrument (Stage 2a) then trains three times, once per mode
(DECISIONS.md #14 on seeds). Probe 3 (blank-image control) exists
specifically because reshaping frequency also reshapes how natural
the text looks — DECISIONS.md #10.

Grapheme clusters, not code points: a Devanagari conjunct is one
visual unit and one vocabulary token for the instrument (DECISIONS.md
#2, #7). Frequency is counted with Unicode grapheme segmentation
(`regex` `\X`), never with `len(text)`.

Algorithm note (DECISIONS.md #28): sentence-level importance sampling
alone cannot hit flattened/inverted targets on a real Indic corpus —
every sentence is already a natural mix, so the convex hull of sentence
bags sits far from uniform/inverted (greedy floor TV ≈ 0.29 / 0.73 on
the 60-line Hindi slice). Flattened/inverted therefore *synthesize*
volume-matched text by (1) allocating an exact integer glyph multiset
from the target PMF and (2) packing those glyphs into sentence-shaped
strings with a bigram-guided walker trained on the source corpus, so
local co-occurrence structure is preserved as far as the quota allows.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import regex

Mode = Literal["natural", "flattened", "inverted"]
MODES: tuple[Mode, ...] = ("natural", "flattened", "inverted")

# Stage 1 acceptance: TV(realized, target) must be ≤ this for every
# mode on a Probe-1-scale corpus. Natural is exact (0); flattened /
# inverted are synthesis-limited but must clear this bar. Defined here
# and mirrored in IMPLEMENTATION.md so the tolerance is not "whatever
# the last smoke run got."
TARGET_TV_TOLERANCE = 0.08

# Whitespace-only clusters are kept in natural text but excluded from
# the frequency target — otherwise spaces dominate every histogram.
_WS = regex.compile(r"^\s+$")

# Probe 1's exposure variable is about *script* glyphs (conjuncts,
# matras), not Latin digits / `%` that show up in GlotOCR sentences.
_INDIC = regex.compile(
    r"[\u0900-\u097F"  # Devanagari
    r"\u0980-\u09FF"  # Bengali
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0C80-\u0CFF"  # Kannada
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u1C50-\u1C7F"  # Ol Chiki (Santhali)
    r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"  # Arabic (Kashmiri)
)


def grapheme_clusters(text: str) -> list[str]:
    """
    Segment text into grapheme clusters (akshara-level for Indic).

    Why `\X` and not `list(text)`: `क्षि` is three code points and one
    visual unit. Counting code points would credit the virama and the
    matra as separate "glyphs," which is exactly the mistake Probe 1
    is designed to avoid. `regex` is required — stdlib `re` does not
    implement `\X`.

    Called from every function in this file and from `render.py` when
    it writes per-cluster ground-truth boxes.
    """
    return regex.findall(r"\X", text)


def glyph_clusters(text: str) -> list[str]:
    """
    Script-bearing grapheme clusters — the support of the frequency dial.

    Drops whitespace and non-Indic characters so the dial cannot waste
    its mass promoting a lone `%` to the frequency of `र`.
    """
    return [
        c for c in grapheme_clusters(text)
        if not _WS.match(c) and _INDIC.search(c)
    ]


def count_clusters(texts: Iterable[str]) -> Counter:
    """Bag-of-clusters over a corpus. Feeds `target_distribution`."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(glyph_clusters(text))
    return counts


def normalize_counts(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def target_distribution(natural_counts: Counter, mode: Mode) -> dict[str, float]:
    """
    Build the probability mass function the resampled corpus should hit.

    `natural`: the empirical distribution — resampling is a shuffle.
    `flattened`: uniform over the observed Indic support. Pure uniform
    used to be unreachable by sentence resampling (DECISIONS.md #25);
    synthesis (#28) makes it the honest flattened target for Probe 1.
    `inverted`: swap probability mass by frequency rank — the rarest
    natural cluster inherits the mass of the most common, and so on.
    That is the Probe 1 starvation condition.
    """
    natural = normalize_counts(natural_counts)
    if not natural:
        return {}
    if mode == "natural":
        return natural

    clusters = list(natural.keys())
    n = len(clusters)

    if mode == "flattened":
        return {c: 1.0 / n for c in clusters}

    ranked = sorted(clusters, key=lambda c: (natural[c], c))
    inv_mass = {ranked[i]: natural[ranked[n - 1 - i]] for i in range(n)}
    z = sum(inv_mass.values())
    return {c: inv_mass[c] / z for c in clusters}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    """
    TV distance between two distributions on a shared alphabet.

    Stage 1 acceptance uses this against TARGET_TV_TOLERANCE. Bounded
    in [0, 1], symmetric, and readable in a histogram report.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def matches_target(realized: dict[str, float], target: dict[str, float],
                   tolerance: float = TARGET_TV_TOLERANCE) -> bool:
    """True iff TV(realized, target) ≤ tolerance — the Stage 1 histogram gate."""
    return total_variation(realized, target) <= tolerance


def target_counts_from_pmf(target: dict[str, float], total: int) -> Counter:
    """
    Integer glyph multiset realizing `target` with exactly `total` draws.

    Largest-remainder method so the counts sum to `total` exactly and
    the induced empirical PMF has the lowest possible discretization
    error for that volume. This is what makes synthesis able to hit
    TV ≪ sentence-resampling floors: the bag is correct by construction
    before any ordering happens.
    """
    if total <= 0 or not target:
        return Counter()
    items = sorted(target.keys())
    raw = {c: target[c] * total for c in items}
    floors = {c: int(np.floor(raw[c])) for c in items}
    used = sum(floors.values())
    remainders = sorted(items, key=lambda c: (raw[c] - floors[c], c), reverse=True)
    for c in remainders[: max(0, total - used)]:
        floors[c] += 1
    # Guard: floating error can leave us one short/long.
    used = sum(floors.values())
    if used < total and remainders:
        floors[remainders[0]] += total - used
    elif used > total:
        for c in sorted(floors, key=floors.get, reverse=True):
            take = min(floors[c], used - total)
            floors[c] -= take
            used -= take
            if used == total:
                break
    return Counter({c: n for c, n in floors.items() if n > 0})


def _bigram_table(sentences: Sequence[str]) -> dict[str, Counter]:
    """
    Left-cluster → next-cluster counts over Indic glyphs only.

    Used by the synthesizer to prefer co-occurrences that existed in
    the source corpus (DECISIONS.md #10), not to invent a full LM.
    """
    table: dict[str, Counter] = {}
    for text in sentences:
        clusters = glyph_clusters(text)
        for i in range(len(clusters) - 1):
            left, right = clusters[i], clusters[i + 1]
            table.setdefault(left, Counter()).update([right])
    return table


def _sentence_initial_counts(sentences: Sequence[str]) -> Counter:
    """First Indic cluster of each source sentence — start-of-line prior."""
    starts: Counter = Counter()
    for text in sentences:
        clusters = glyph_clusters(text)
        if clusters:
            starts.update([clusters[0]])
    return starts


def _sample_from_counter(weights: Counter, rng: np.random.Generator) -> str | None:
    """Draw one key proportional to positive weights; None if empty."""
    items = [(k, w) for k, w in weights.items() if w > 0]
    if not items:
        return None
    keys, vals = zip(*items)
    p = np.asarray(vals, dtype=np.float64)
    p = p / p.sum()
    return keys[int(rng.choice(len(keys), p=p))]


def _next_glyph_weights(
    prev: str | None,
    remaining: Counter,
    bigrams: dict[str, Counter],
    unigram: Counter,
) -> Counter:
    """
    Score remaining glyphs for the next slot.

    Prefer bigram continuation when the left context is known, else the
    corpus unigram, always masked by remaining quota. Quota is hard —
    a zero-remaining glyph cannot be chosen — which is what keeps the
    realized bag equal to the target bag.
    """
    scores: Counter = Counter()
    if prev is not None and prev in bigrams:
        for g, w in bigrams[prev].items():
            if remaining[g] > 0:
                scores[g] = w * remaining[g]
    if not scores:
        for g, n in remaining.items():
            if n > 0:
                # Unigram prior × remaining, with a floor so hapaxes
                # still get placed when the bigram table has no advice.
                scores[g] = max(unigram.get(g, 0), 1) * n
    return scores


def _pack_sentence(
    remaining: Counter,
    length: int,
    bigrams: dict[str, Counter],
    unigram: Counter,
    starts: Counter,
    rng: np.random.Generator,
) -> str:
    """
    Consume up to `length` glyphs from `remaining`, ordered by bigram.

    Inserts a space every 2–4 glyphs so the renderer can wrap on
    whitespace the way it does for natural text. Spaces are not part
    of the frequency dial (see `glyph_clusters`).
    """
    if length <= 0 or not remaining:
        return ""
    start_weights = Counter(
        {g: starts.get(g, 0) * remaining[g] for g in remaining if remaining[g] > 0}
    )
    if sum(start_weights.values()) == 0:
        start_weights = Counter({g: remaining[g] for g in remaining if remaining[g] > 0})
    first = _sample_from_counter(start_weights, rng)
    if first is None:
        return ""
    remaining[first] -= 1
    if remaining[first] <= 0:
        del remaining[first]
    tokens = [first]
    prev = first
    for i in range(1, length):
        weights = _next_glyph_weights(prev, remaining, bigrams, unigram)
        nxt = _sample_from_counter(weights, rng)
        if nxt is None:
            break
        remaining[nxt] -= 1
        if remaining[nxt] <= 0:
            del remaining[nxt]
        tokens.append(nxt)
        prev = nxt
    # Chunk into pseudo-words for wrap-friendly output.
    words: list[str] = []
    i = 0
    while i < len(tokens):
        span = int(rng.integers(2, 5))
        words.append("".join(tokens[i: i + span]))
        i += span
    return " ".join(words)


def synthesize_toward_target(
    natural_sentences: Sequence[str],
    target: dict[str, float],
    *,
    total_glyphs: int | None = None,
    n_sentences: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """
    Build volume-matched text whose Indic-cluster bag matches `target`.

    Why synthesis rather than sentence resampling: on the Hindi
    GlotOCR slice, a greedy oracle that only picks existing sentences
    cannot get below TV ≈ 0.29 (flattened) / 0.73 (inverted) — every
    sentence is mostly common glyphs. Probe 1 needs the realized
    histogram to *be* the experimental variable, so we allocate the
    glyph multiset first and only then impose local LM structure via
    bigrams. Naturalness is degraded relative to pure resampling;
    Probe 3 exists to measure that confound (DECISIONS.md #10, #28).

    Called from `resample_corpus` for `flattened` and `inverted`.
    """
    rng = rng or np.random.default_rng()
    natural_counts = count_clusters(natural_sentences)
    if total_glyphs is None:
        total_glyphs = int(sum(natural_counts.values()))
    if n_sentences is None:
        n_sentences = max(1, len([s for s in natural_sentences if glyph_clusters(s)]))

    quota = target_counts_from_pmf(target, total_glyphs)
    # Copy — pack mutates in place.
    remaining = Counter(quota)
    bigrams = _bigram_table(natural_sentences)
    unigram = Counter(natural_counts)
    starts = _sentence_initial_counts(natural_sentences)

    # Per-sentence lengths ≈ source length distribution, clipped so
    # the last sentence can absorb the remainder.
    src_lens = [max(1, len(glyph_clusters(s))) for s in natural_sentences if glyph_clusters(s)]
    if not src_lens:
        src_lens = [12]
    mean_len = max(1, int(round(np.mean(src_lens))))

    texts: list[str] = []
    for si in range(n_sentences):
        left = sum(remaining.values())
        if left <= 0:
            break
        if si == n_sentences - 1:
            length = left
        else:
            # Leave at least one glyph per remaining sentence slot.
            slots_after = n_sentences - si - 1
            length = min(mean_len, max(1, left - slots_after))
            length = int(np.clip(length, 1, left))
        texts.append(
            _pack_sentence(remaining, length, bigrams, unigram, starts, rng)
        )
    # Any leftover (rounding / early empty starts) — dump into one more line.
    if sum(remaining.values()) > 0:
        texts.append(
            _pack_sentence(
                remaining, sum(remaining.values()), bigrams, unigram, starts, rng
            )
        )
    return [t for t in texts if t]


@dataclass
class ResampleResult:
    """Resampled/synthesized sentences plus Stage 1 histogram diagnostics."""

    texts: list[str]
    mode: Mode
    natural: dict[str, float]
    target: dict[str, float]
    realized: dict[str, float]
    tv_distance: float

    def within_tolerance(self, tolerance: float = TARGET_TV_TOLERANCE) -> bool:
        """Stage 1 acceptance gate for this result."""
        return self.tv_distance <= tolerance

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tv_distance": self.tv_distance,
            "within_tolerance": self.within_tolerance(),
            "tolerance": TARGET_TV_TOLERANCE,
            "n_texts": len(self.texts),
            "natural": self.natural,
            "target": self.target,
            "realized": self.realized,
            "texts": self.texts,
        }


def resample_corpus(
    texts: Sequence[str],
    mode: Mode,
    rng: np.random.Generator | None = None,
    n_out: int | None = None,
    repair: bool = True,  # kept for API compat; ignored — synthesis replaced repair
) -> ResampleResult:
    """
    Produce a corpus whose realized glyph-cluster frequencies track
    `target_distribution(..., mode)` within TARGET_TV_TOLERANCE.

    `natural`: shuffle (or with-replacement sample if n_out differs) —
    exact multiset when volumes match.
    `flattened` / `inverted`: bigram-guided synthesis of a target
    glyph bag (DECISIONS.md #28). The unused `repair` flag remains so
    older call sites do not break; repair-vs-IS was the previous
    inverted path and is no longer used.

    `n_out` defaults to `len(texts)` so Probe 1 conditions stay
    sentence-count-matched; glyph volume is matched to the source
    total independently, which is what exposure actually counts.
    """
    del repair  # API compat only
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    rng = rng or np.random.default_rng()
    sentences = [t for t in texts if t and glyph_clusters(t)]
    if not sentences:
        return ResampleResult([], mode, {}, {}, {}, 0.0)

    natural_counts = count_clusters(sentences)
    natural = normalize_counts(natural_counts)
    target = target_distribution(natural_counts, mode)
    n_out = len(sentences) if n_out is None else int(n_out)

    if mode == "natural":
        if n_out == len(sentences):
            picked = list(sentences)
            rng.shuffle(picked)
        else:
            idx = rng.choice(len(sentences), size=n_out, replace=True)
            picked = [sentences[i] for i in idx]
        realized = normalize_counts(count_clusters(picked))
        return ResampleResult(
            texts=picked,
            mode=mode,
            natural=natural,
            target=target,
            realized=realized,
            tv_distance=total_variation(realized, target),
        )

    picked = synthesize_toward_target(
        sentences,
        target,
        total_glyphs=int(sum(natural_counts.values())),
        n_sentences=n_out,
        rng=rng,
    )
    realized = normalize_counts(count_clusters(picked))
    return ResampleResult(
        texts=picked,
        mode=mode,
        natural=natural,
        target=target,
        realized=realized,
        tv_distance=total_variation(realized, target),
    )


def save_resample_result(result: ResampleResult, path: Path | str) -> Path:
    """Checkpoint a resampled corpus. Probe 1 training reads this, not the raw source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_resample_result(path: Path | str) -> ResampleResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResampleResult(
        texts=payload["texts"],
        mode=payload["mode"],
        natural=payload["natural"],
        target=payload["target"],
        realized=payload["realized"],
        tv_distance=float(payload["tv_distance"]),
    )


def histogram_report(result: ResampleResult, top_k: int = 12) -> str:
    """
    Human-readable Stage 1 acceptance check: realized vs target for
    the clusters that moved the most, plus the tolerance gate.
    """
    deltas = sorted(
        result.target.keys(),
        key=lambda c: abs(result.realized.get(c, 0.0) - result.target[c]),
        reverse=True,
    )[:top_k]
    gate = "PASS" if result.within_tolerance() else "FAIL"
    lines = [
        f"mode={result.mode}  n={len(result.texts)}  "
        f"TV={result.tv_distance:.4f}  tolerance={TARGET_TV_TOLERANCE}  [{gate}]",
        f"{'cluster':<12} {'natural':>8} {'target':>8} {'realized':>8}",
    ]
    for c in deltas:
        lines.append(
            f"{c!r:<12} {result.natural.get(c, 0):8.4f} "
            f"{result.target.get(c, 0):8.4f} {result.realized.get(c, 0):8.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Smoke check on GlotOCR Hindi ground truth if present."""
    gt = Path(__file__).resolve().parents[2] / "data" / "raw" / "hindi" / "ground_truth.jsonl"
    if not gt.exists():
        print(f"no corpus at {gt}; pass texts programmatically")
        return
    texts = [json.loads(line)["text"] for line in gt.read_text(encoding="utf-8").splitlines()]
    rng = np.random.default_rng(0)
    for mode in MODES:
        result = resample_corpus(texts, mode, rng=rng)
        print(histogram_report(result))
        print()


if __name__ == "__main__":
    main()

"""
Grapheme-cluster tokenizer for the instrument model (Stage 2a).

Why grapheme clusters and not BPE: see DECISIONS.md #2. Probe 1 needs
to measure exposure per VISUAL unit -- BPE's frequency-driven merges
would tangle "how often the vision encoder saw this glyph" with "how
the tokenizer happened to segment it," two different notions of
frequency. Grapheme clusters keep that variable clean.

Built fresh per training run, not loaded from a fixed universal
vocabulary -- Probe 1 trains on Devanagari only (DECISIONS.md #6), so
the vocabulary for that instrument is built from whatever corpus that
specific run actually uses, per docs/stage2_design_notes.md.
"""

import json
import regex  # NOT the stdlib `re` -- only `regex` supports \X (grapheme cluster boundaries)
from collections import Counter

PAD, BOS, EOS, RARE = "<PAD>", "<BOS>", "<EOS>", "<RARE>"
SPECIAL_TOKENS = [PAD, BOS, EOS, RARE]


def split_graphemes(text: str) -> list[str]:
    """
    Splits a string into grapheme clusters -- the visually-atomic
    units (base consonant + matras + virama + ZWJ, treated as one).
    \\X is regex's Unicode grapheme cluster boundary, NOT available in
    Python's stdlib re module. Used everywhere in this repo that needs
    to align or count text at the "one visual character" level rather
    than the "one Unicode code point" level (see DECISIONS.md #7).
    """
    return regex.findall(r"\X", text)


class GraphemeTokenizer:
    """
    Maps grapheme clusters <-> integer ids for one training run's
    corpus. Not shared across runs with different corpora -- each
    instrument run (per Probe 1 condition + seed) may see a slightly
    different realized vocabulary, since the renderer's resampling
    changes which clusters appear and how often (DECISIONS.md #29).
    """

    def __init__(self):
        self.cluster_to_id: dict[str, int] = {}
        self.id_to_cluster: dict[int, str] = {}

    def build_vocab(self, texts: list[str], min_freq: int = 5) -> None:
        """
        Counts every grapheme cluster across `texts`, keeps ones
        appearing at least `min_freq` times, assigns ids. Special
        tokens always occupy ids 0-3 so they're stable across runs
        even though the rest of the vocabulary isn't.

        Called once, at the start of a training run, over that run's
        full training corpus (the renderer's rendered-page ground
        truth text, not the raw pre-resampling source text).
        """
        counts = Counter()
        for text in texts:
            counts.update(split_graphemes(text))

        # special tokens first, fixed ids
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.cluster_to_id[tok] = i
            self.id_to_cluster[i] = tok

        next_id = len(SPECIAL_TOKENS)
        # sorted by frequency descending, then alphabetically, for
        # reproducibility -- two runs over the same corpus should
        # produce the same vocabulary, not one that depends on dict
        # iteration order.
        for cluster, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if count < min_freq:
                continue  # below floor -> falls back to <RARE> at encode time
            self.cluster_to_id[cluster] = next_id
            self.id_to_cluster[next_id] = cluster
            next_id += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Text -> list of ids. Unknown/rare clusters map to <RARE>
        rather than raising -- an instrument that's never seen a
        cluster should still be able to attempt a prediction (this
        matters directly for Probe 5b, the zero-shot floor test on
        Santhali/Kashmiri, where EVERY cluster is unseen by
        construction).
        """
        rare_id = self.cluster_to_id[RARE]
        ids = [self.cluster_to_id.get(c, rare_id) for c in split_graphemes(text)]
        if add_special_tokens:
            ids = [self.cluster_to_id[BOS]] + ids + [self.cluster_to_id[EOS]]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Ids -> text. Used to read out what the model actually generated during eval/probes."""
        clusters = [self.id_to_cluster.get(i, RARE) for i in ids]
        if strip_special:
            clusters = [c for c in clusters if c not in SPECIAL_TOKENS]
        return "".join(clusters)

    def __len__(self) -> int:
        return len(self.cluster_to_id)

    def save(self, path: str) -> None:
        """Persists the vocabulary so a resumed training run (per AGENTS.md's resumability standard) uses the SAME ids, not a re-built vocabulary that might differ."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cluster_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "GraphemeTokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            tok.cluster_to_id = json.load(f)
        tok.id_to_cluster = {v: k for k, v in tok.cluster_to_id.items()}
        return tok


if __name__ == "__main__":
    # Smoke test: build a tiny vocab, round-trip encode/decode.
    corpus = [
        "हिन्दी एक भाषा है।",
        "यह एक वाक्य है।",
        "भाषा सीखना अच्छा है।",
    ] * 5  # repeat so common clusters clear the min_freq=5 floor in this tiny example

    tok = GraphemeTokenizer()
    tok.build_vocab(corpus, min_freq=5)
    print(f"vocab size: {len(tok)}")

    sample = "यह भाषा है।"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(f"original: {sample}")
    print(f"ids:      {ids}")
    print(f"decoded:  {decoded}")
    assert decoded == sample, f"round-trip mismatch: {decoded!r} != {sample!r}"
    print("round-trip OK")

    # unseen cluster -> should fall back to <RARE>, not crash
    unseen = tok.encode("कश्मीर")  # clusters likely below the tiny corpus's min_freq
    print(f"unseen text encoded without crashing: {unseen}")

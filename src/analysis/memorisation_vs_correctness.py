"""
Split Probe 5 synthetic-natural lines by verbatim overlap with training.

Why this exists: the paper's AUROC 0.838 (docs/paper_defensibility_stats.md
Offline Analysis 9 / Follow-Up 3) is computed on Probe 5 Hindi natural
predictions. Those evaluation lines are *not* held out from the training
manifest. If every eval string is also a training string, that AUROC
cannot distinguish "confidence tracks correctness" from "confidence
tracks memorised text." This script does the split on already-committed
jsonl — no new inference.

Called from: `python src/analysis/memorisation_vs_correctness.py`
Output: `docs/memorisation_vs_correctness.md`
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from paper_defensibility_stats import auroc, load_jsonl  # noqa: E402

PROBE5 = "data/probe_results/probe5_hindi_natural_seed{}.jsonl"
MANIFEST = "data/manifests/hindi_natural.jsonl"
GT_LIKELIHOOD = "data/probe_results/probe_gt_likelihood_hindi_natural_seed0.jsonl"


def split_records(records: list[dict], train_texts: set[str]) -> tuple[list[dict], list[dict]]:
    """Partition Probe 5 records by exact `ground_truth` ∈ training `text`."""
    matching, nonmatching = [], []
    for r in records:
        gt = r.get("ground_truth") or ""
        if gt in train_texts:
            matching.append(r)
        else:
            nonmatching.append(r)
    return matching, nonmatching


def acc_auroc(rows: list[dict]) -> tuple[int, float, float]:
    """n, line accuracy, Mann–Whitney AUROC on (confidence, correct)."""
    if not rows:
        return 0, float("nan"), float("nan")
    labels = [int(bool(r["correct"])) for r in rows]
    scores = [float(r["confidence"]) for r in rows]
    acc = sum(labels) / len(labels)
    return len(rows), acc, auroc(scores, labels)


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.4f}"


def gt_likelihood_overlap(repo: Path, train_texts: set[str]) -> dict:
    """
    Different eval set (60 real-scan teacher-forced strings), for context.

    Not the Probe 5 split. README already notes overlap on this pool;
    recompute from committed jsonl rather than restating a remembered count.
    """
    path = repo / GT_LIKELIHOOD
    if not path.exists():
        return {"n": 0, "n_match": 0, "n_unique": 0}
    rows = [r for r in load_jsonl(path) if r.get("condition") == "real"]
    gts = [r.get("ground_truth") or "" for r in rows]
    unique = set(gts)
    return {
        "n": len(gts),
        "n_unique": len(unique),
        "n_match": sum(1 for t in unique if t in train_texts),
    }


def write_doc(repo: Path) -> str:
    manifest = load_jsonl(repo / MANIFEST)
    train_texts = {r["text"] for r in manifest if "text" in r}

    lines = [
        "# Memorisation vs correctness (Probe 5 AUROC split)",
        "",
        "Question: does the synthetic-natural AUROC in",
        "`docs/paper_defensibility_stats.md` (pooled **0.8381**, cite that",
        "file — do not treat this page as a second copy of Follow-Up 3)",
        "rank genuine correctness, or only whether the line was in training?",
        "",
        "## Method",
        "",
        "- Training strings: exact `text` field of",
        f"  `{MANIFEST}` (n={len(manifest)} rows, {len(train_texts)} unique `text` values).",
        "- Eval: `records[]` in committed",
        "  `data/probe_results/probe5_hindi_natural_seed{0,1,2}.jsonl`.",
        "- Match rule: `ground_truth ==` some training `text` (verbatim;",
        "  no NFC extra pass, no grapheme rewrite).",
        "- Metrics: line accuracy (`correct` already uses Tier 1/2 in Probe 5)",
        "  and Mann–Whitney AUROC (`paper_defensibility_stats.auroc`) on",
        "  `(confidence, correct)`. Same estimator as Follow-Up 3.",
        "- No new model forwards. Producer of the jsonl:",
        "  `src/probes/probe5_calibration.py` samples",
        "  `random.Random(0).sample(manifest_rows, n_samples)` from this",
        "  same manifest, then scores the trained instrument.",
        "",
        "| Seed | subset | n | accuracy | AUROC |",
        "|---|---|---:|---:|---:|",
    ]

    pooled_match: list[dict] = []
    pooled_non: list[dict] = []
    pooled_all: list[dict] = []

    for s in range(3):
        path = repo / PROBE5.format(s)
        rows = load_jsonl(path)
        matching, nonmatching = split_records(rows, train_texts)
        pooled_match.extend(matching)
        pooled_non.extend(nonmatching)
        pooled_all.extend(rows)
        for name, subset in (
            ("all", rows),
            ("in training manifest", matching),
            ("not in training manifest", nonmatching),
        ):
            n, acc, auc = acc_auroc(subset)
            lines.append(f"| {s} | {name} | {n} | {fmt(acc)} | {fmt(auc)} |")

    lines.append("")
    lines.append("### Pooled (seeds 0–2 concatenated)")
    lines.append("")
    lines.append("| subset | n | accuracy | AUROC |")
    lines.append("|---|---:|---:|---:|")
    for name, subset in (
        ("all", pooled_all),
        ("in training manifest", pooled_match),
        ("not in training manifest", pooled_non),
    ):
        n, acc, auc = acc_auroc(subset)
        lines.append(f"| {name} | {n} | {fmt(acc)} | {fmt(auc)} |")

    n_all, acc_all, auc_all = acc_auroc(pooled_all)
    n_non, _, auc_non = acc_auroc(pooled_non)

    lines += [
        "",
        "## Finding",
        "",
    ]
    if n_non == 0:
        lines += [
            "**The non-matching subset is empty.** Every Probe 5 synthetic-natural",
            f"evaluation line (n={n_all} across three seeds) has `ground_truth`",
            "verbatim in `hindi_natural.jsonl`. That is expected from the",
            "sampler: Probe 5 draws its eval rows *from* the training manifest",
            "(`probe5_calibration.py` `run_probe5`, `Random(0).sample`).",
            "",
            "Pooled AUROC on the matching (i.e. only) subset is therefore the",
            f"same computation as Follow-Up 3: {fmt(auc_all)} (accuracy {fmt(acc_all)}).",
            "AUROC on held-out-from-manifest strings **cannot be computed**",
            "from these files. The 0.838 figure is **not** evidence that",
            "confidence tracks correctness on unseen text. It is also not a",
            "demonstration that AUROC *collapses* off-manifest — there is no",
            "off-manifest Probe 5 slice to collapse.",
            "",
            "Settling the exclusivity claim needs a Probe 5 rerun whose eval",
            "lines are disjoint from the training `text` set (or a different",
            "already-scored jsonl with that property). That rerun is **not**",
            "this script.",
        ]
    elif math.isnan(auc_non):
        lines += [
            "The non-matching subset is non-empty but AUROC is undefined",
            "(one class missing). Report the table above; do not fill a number.",
        ]
    else:
        lines += [
            f"Non-matching n={n_non}, AUROC={fmt(auc_non)} vs matching",
            f"AUROC={fmt(acc_auroc(pooled_match)[2])}.",
            "If non-matching AUROC holds near the pooled figure, confidence",
            "is tracking correctness beyond verbatim memorisation. If it",
            "falls to chance (~0.5), the headline AUROC was tracking overlap.",
        ]

    extra = gt_likelihood_overlap(repo, train_texts)
    lines += [
        "",
        "## Not this eval: 60-image GT-likelihood pool",
        "",
        "Teacher-forced Table 6 uses real scans, not Probe 5 crops. Unique",
        f"GT strings in `{GT_LIKELIHOOD}` condition=real:",
        f"{extra['n_unique']} unique / {extra['n']} rows;",
        f"**{extra['n_match']}** of those unique strings appear verbatim in",
        "`hindi_natural.jsonl`. That overlap is a contamination note for",
        "Section 8 / n-gram splits, not a substitute for the Probe 5 AUROC",
        "split above.",
        "",
        "## Reproduce",
        "",
        "```",
        "PYTHONPATH=src python src/analysis/memorisation_vs_correctness.py",
        "```",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out = repo / "docs" / "memorisation_vs_correctness.md"
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    text = write_doc(_ROOT)
    print(text, end="")
    print(f"[memorisation] wrote {_ROOT / 'docs' / 'memorisation_vs_correctness.md'}")


if __name__ == "__main__":
    main()

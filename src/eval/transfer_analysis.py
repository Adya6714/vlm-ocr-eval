"""
src/eval/transfer_analysis.py

Stage 5b — pre-specified rank-correlation transfer (DECISIONS.md #89).

Why this file exists
--------------------
IMPLEMENTATION.md Stage 5b asked for a rank-correlation test with a
permutation null *before* anyone looks at the coefficient. Decision #89
locks the statistic (Spearman), the two vectors, the null, and the
primary cohort. This module is the only place that coefficient is
allowed to be computed, so the choice cannot drift into an after-the-fact
Kendall/Pearson hunt.

It does not call the Sarvam API. Paid pages are fetched by
`src/probes/stage5b_sarvam_pages.py`, which refuses to POST without an
exact rupee confirmation. Stage 6 (`src/probes/cascade.py`) reads the
same paired table this module builds.

Called from: CLI below; tests/test_transfer_analysis.py (synthetic
vectors only — tests must not peek at the live probe jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SRC / "analysis") not in sys.path:
    sys.path.insert(0, str(_SRC / "analysis"))
if str(_SRC / "eval") not in sys.path:
    sys.path.insert(0, str(_SRC / "eval"))

from analysis.grapheme_cer_rescore import cer_and_accuracy  # noqa: E402
from analysis.paper_defensibility_stats import spearman  # noqa: E402

# --- Decision #89 constants. Do not change after looking at ρ. ---
N_PERM = 10_000
PERM_SEED = 0
PRICE_INR_PER_PAGE = 0.5
STATISTIC_NAME = "Spearman rho"
INSTRUMENT_DIFFICULTY = (
    "negative mean over seeds of teacher-forced mean_log_p_gt "
    "on condition=real (higher = the instrument finds the page harder)"
)
PRODUCTION_ERROR = "Tier-1 grapheme-cluster CER of the production engine vs GT"


def load_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file, skipping blank lines. Used by 5b pairing and Stage 6."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_cer(pred: str, gt: str) -> float:
    """
    Grapheme CER with empty-GT inf collapsed to 1.0.

    Why collapse: Stage 5b's primary cohort is real text lines, not
    blanks. If a stray empty GT appears, inf would poison Spearman.
    """
    d = cer_and_accuracy(pred or "", gt or "")
    cer = d["cer"]
    if cer == float("inf") or math.isnan(cer):
        return 1.0
    return float(cer)


def instrument_difficulty_by_id(
    repo_root: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, float]:
    """
    Per-image instrument difficulty from committed GT-likelihood jsonl.

    Why mean_log_p_gt and not Probe 5b mean_confidence: Decision #89
    ranks pages by how little probability the instrument assigns the
    ground-truth token sequence (teacher-forced). Confidence of the
    *generated* string is the Stage 6 routing score, a different
    estimand.

    Returns image_id -> −mean_s mean_log_p_gt. Missing ids omitted.
    """
    per: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        path = (
            repo_root
            / "data"
            / "probe_results"
            / f"probe_gt_likelihood_hindi_natural_seed{seed}.jsonl"
        )
        for row in load_jsonl(path):
            if row.get("condition") != "real":
                continue
            lp = row.get("mean_log_p_gt")
            if lp is None:
                continue
            per[str(row["image_id"])].append(float(lp))
    out = {}
    for iid, vals in per.items():
        if len(vals) != len(seeds):
            continue
        out[iid] = -float(np.mean(vals))
    return out


def instrument_confidence_by_id(
    repo_root: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    condition: str = "hindi",
) -> dict[str, float]:
    """
    Per-image Probe 5b mean_confidence, averaged across seeds.

    Stage 6's router uses this, not GT log-likelihood: escalation is a
    decision the model can make at inference time without seeing GT.
    """
    per: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        path = (
            repo_root
            / "data"
            / "probe_results"
            / f"probe5b_hindi_natural_seed{seed}.jsonl"
        )
        for row in load_jsonl(path):
            if row.get("condition") != condition:
                continue
            c = row.get("mean_confidence")
            if c is None:
                continue
            per[str(row["image_id"])].append(float(c))
    return {iid: float(np.mean(vals)) for iid, vals in per.items() if vals}


def hindi_probe5b_ids(repo_root: Path) -> list[str]:
    """The 60 Hindi real image_ids in Probe 5b seed 0 (canonical pool)."""
    path = repo_root / "data/probe_results/probe5b_hindi_natural_seed0.jsonl"
    ids = []
    seen = set()
    for row in load_jsonl(path):
        if row.get("condition") != "hindi":
            continue
        iid = str(row["image_id"])
        if iid not in seen:
            seen.add(iid)
            ids.append(iid)
    return ids


def parse_id_variant(image_path: str) -> tuple[str, str] | None:
    """Split `137_plain.png` / `137_degraded.png` into (id, variant)."""
    name = Path(image_path).name
    stem = name.rsplit(".", 1)[0]
    for variant in ("plain", "degraded"):
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], variant
    return None


def load_engine_rows(
    repo_root: Path, engine: str, language: str = "hindi"
) -> dict[tuple[str, str], dict]:
    """Map (image_id, variant) -> prediction row for a Stage 0 engine."""
    path = repo_root / "data" / "predictions" / engine / f"{language}.jsonl"
    out: dict[tuple[str, str], dict] = {}
    for row in load_jsonl(path):
        parsed = parse_id_variant(row.get("image_path") or "")
        if parsed is None:
            continue
        out[parsed] = row
    return out


def load_sarvam_rows(paths: list[Path]) -> list[dict]:
    """Concat Stage 5a + 5b jsonl; later files override same (id, path)."""
    by_key: dict[tuple[str, str], dict] = {}
    for path in paths:
        for row in load_jsonl(path):
            iid = str(row.get("image_id", ""))
            ip = str(row.get("image_path", ""))
            by_key[(iid, ip)] = row
    return list(by_key.values())


def spearman_perm_p(
    x: list[float],
    y: list[float],
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> dict:
    """
    Spearman ρ and a two-sided permutation p-value.

    Null: y labels are exchangeable; shuffle y, keep x. p-value is
    (1 + #{|ρ*| ≥ |ρ_obs|}) / (1 + n_perm) so it cannot be exactly 0.
    Pre-registered in Decision #89; n_perm and seed are not knobs.
    """
    if len(x) != len(y) or len(x) < 3:
        return {
            "rho": float("nan"),
            "p_value": float("nan"),
            "n": len(x),
            "n_perm": n_perm,
            "seed": seed,
        }
    rho = spearman(x, y)
    if rho != rho:  # NaN: a ranking was constant
        return {
            "rho": float("nan"),
            "p_value": float("nan"),
            "n": len(x),
            "n_perm": n_perm,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float)
    x_list = list(x)
    thresh = abs(rho)
    count = 0
    for _ in range(n_perm):
        y_star = rng.permutation(y_arr)
        rho_star = spearman(x_list, y_star.tolist())
        if rho_star != rho_star:
            continue
        if abs(rho_star) >= thresh:
            count += 1
    p = (1.0 + count) / (1.0 + n_perm)
    return {
        "rho": float(rho),
        "p_value": float(p),
        "n": len(x),
        "n_perm": n_perm,
        "seed": seed,
        "n_extreme": count,
    }


def pair_sarvam_cohort(
    repo_root: Path,
    sarvam_rows: list[dict],
    *,
    script: str = "hindi",
    variant: str = "plain",
) -> list[dict]:
    """
    One row per image that has instrument difficulty and a Sarvam text.

    variant `plain` matches Probe 5b / GT-likelihood paths (`*_plain.png`).
    `degraded` is Decision #15's Tier B arm (only if those pages were fetched).
    """
    difficulty = instrument_difficulty_by_id(repo_root)
    suffix = f"_{variant}.png"
    paired = []
    for row in sarvam_rows:
        if row.get("script") != script:
            continue
        if row.get("error"):
            continue
        ip = row.get("image_path") or ""
        if not ip.endswith(suffix):
            # 5a used img_plain_path which already ends in _plain.png
            if variant == "plain" and "_plain" not in Path(ip).name:
                continue
            if variant == "degraded" and "_degraded" not in Path(ip).name:
                continue
        iid = str(row["image_id"])
        if iid not in difficulty:
            continue
        gt = row.get("ground_truth") or ""
        pred = row.get("sarvam_text") or ""
        paired.append(
            {
                "image_id": iid,
                "script": script,
                "variant": variant,
                "instrument_difficulty": difficulty[iid],
                "sarvam_cer": finite_cer(pred, gt),
                "sarvam_confidence": row.get("confidence"),
                "ground_truth": gt,
            }
        )
    paired.sort(key=lambda r: int(r["image_id"]) if str(r["image_id"]).isdigit() else 0)
    return paired


def pair_engine_cohort(
    repo_root: Path,
    engine: str,
    image_ids: list[str],
    *,
    variant: str = "plain",
    language: str = "hindi",
) -> list[dict]:
    """Instrument difficulty vs a local engine's grapheme CER on the same ids."""
    difficulty = instrument_difficulty_by_id(repo_root)
    engine_rows = load_engine_rows(repo_root, engine, language)
    gt_by_id = {}
    gt_path = repo_root / "data" / "raw" / language / "ground_truth.jsonl"
    for row in load_jsonl(gt_path):
        gt_by_id[str(row["id"])] = row.get("text") or ""
    paired = []
    for iid in image_ids:
        if iid not in difficulty:
            continue
        rec = engine_rows.get((iid, variant))
        if rec is None:
            continue
        gt = gt_by_id.get(iid, "")
        pred = rec.get("predicted_text") or ""
        paired.append(
            {
                "image_id": iid,
                "engine": engine,
                "variant": variant,
                "instrument_difficulty": difficulty[iid],
                "engine_cer": finite_cer(pred, gt),
                "engine_confidence": rec.get("confidence"),
            }
        )
    return paired


def format_result_block(title: str, result: dict) -> str:
    """Markdown block for ρ, permutation p, n. Used by the 5b report."""
    rho = result["rho"]
    p = result["p_value"]
    rho_s = "nan" if rho != rho else f"{rho:.4f}"
    p_s = "nan" if p != p else f"{p:.4f}"
    return (
        f"### {title}\n\n"
        f"- Statistic: {STATISTIC_NAME} (Decision #89)\n"
        f"- ρ = **{rho_s}**\n"
        f"- permutation p (two-sided, {result['n_perm']} shuffles, "
        f"seed {result['seed']}) = **{p_s}**\n"
        f"- n = **{result['n']}**\n"
    )


def write_stage5b_report(
    repo_root: Path,
    out_path: Path,
    *,
    compute: bool,
) -> str:
    """
    Write docs/stage5b_rank_correlation.md.

    If compute is False, only the pre-registered protocol is written —
    used before spend confirmation so we never look at ρ first.
    """
    lines = [
        "# Stage 5b — rank-correlation transfer\n",
        "Pre-registered in `DECISIONS.md` **#89**. Do not swap Spearman",
        "for Kendall/Pearson after seeing a coefficient.\n",
        "## Protocol (locked before looking at ρ)\n",
        f"- **Statistic:** {STATISTIC_NAME} between instrument difficulty",
        f"  ({INSTRUMENT_DIFFICULTY}) and production error ({PRODUCTION_ERROR}).",
        "- **Sign:** both axes increase with hardness, so a useful ranking",
        "  transfer is **positive** ρ.",
        f"- **Null:** shuffle production CER, {N_PERM} permutations, RNG seed",
        f"  {PERM_SEED}, two-sided p = (1 + #{{|ρ*| ≥ |ρ_obs|}}) / (1 + N).",
        "- **Primary:** Hindi, same `image_id`, Tier A (`*_plain.png`), Sarvam.",
        "- **Secondary (Decision #15):** same Hindi ids on Tier B degraded,",
        "  only if those Extract pages exist in cache/jsonl.",
        "- **Secondary (no extra API):** Tesseract and Surya CER on the same",
        "  Hindi plains. PaddleOCR is listed the same way; it is not a",
        "  matched instrument (Decision #86).",
        "- **Not primary:** pooling Hindi+Santhali+Kashmiri. Language is a",
        "  confounder of image difficulty. Stratified n=10 arms from Stage 5a",
        "  may be reported separately and are underpowered.",
        "- **Not this probe:** per-glyph-class error rates in the original",
        "  IMPLEMENTATION.md 5b bullet. This session's unit is per-image",
        "  (explicit override; see #88).",
        "",
        "## Spend\n",
        "Stage 5a already spent ₹17.50 (35 pages). New Extract calls are",
        "gated by `src/probes/stage5b_sarvam_pages.py --i-confirm-spend-inr`.",
        "At ₹0.5/page: 50 remaining Hindi plains = **₹25.00**; 60 Hindi",
        "degraded = **₹30.00**; both = **₹55.00** (110 pages).",
        "",
    ]
    if not compute:
        lines += [
            "## Results\n",
            "**Not computed.** Spend has not been confirmed, and this",
            "report must not contain a coefficient until the user confirms",
            "the page set (including the ₹0 / n=10 cache-only option).\n",
        ]
        text = "\n".join(lines)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return text

    sarvam_paths = [
        repo_root / "data/probe_results/sarvam_transfer_probe.jsonl",
        repo_root / "data/probe_results/sarvam_stage5b_pages.jsonl",
    ]
    sarvam_rows = load_sarvam_rows(sarvam_paths)
    hindi_ids = hindi_probe5b_ids(repo_root)

    lines.append("## Results\n")
    primary = pair_sarvam_cohort(repo_root, sarvam_rows, script="hindi", variant="plain")
    primary_res = spearman_perm_p(
        [r["instrument_difficulty"] for r in primary],
        [r["sarvam_cer"] for r in primary],
    )
    lines.append(format_result_block("Primary: Hindi plains, Sarvam CER", primary_res))
    lines.append(
        f"Ids: {', '.join(r['image_id'] for r in primary) if primary else '(none)'}\n"
    )

    tier_b = pair_sarvam_cohort(
        repo_root, sarvam_rows, script="hindi", variant="degraded"
    )
    if len(tier_b) >= 3:
        tb_res = spearman_perm_p(
            [r["instrument_difficulty"] for r in tier_b],
            [r["sarvam_cer"] for r in tier_b],
        )
        lines.append(format_result_block("Secondary: Hindi degraded, Sarvam CER", tb_res))
    else:
        lines.append(
            "### Secondary: Hindi degraded, Sarvam CER\n\n"
            f"Not run (n={len(tier_b)}; need Extract on `*_degraded.png`).\n"
        )

    for engine in ("tesseract", "surya", "paddleocr"):
        eng = pair_engine_cohort(repo_root, engine, hindi_ids, variant="plain")
        # Restrict local-engine secondary to the same ids as the Sarvam primary
        # when the Sarvam primary is the spend-limited subset; if Sarvam n==60
        # this is the full pool.
        if primary:
            primary_ids = {r["image_id"] for r in primary}
            if len(primary) < len(hindi_ids):
                eng = [r for r in eng if r["image_id"] in primary_ids]
        res = spearman_perm_p(
            [r["instrument_difficulty"] for r in eng],
            [r["engine_cer"] for r in eng],
        )
        lines.append(
            format_result_block(
                f"Secondary (no API): Hindi plains, {engine} CER", res
            )
        )

    for script in ("santhali", "kashmiri"):
        # These scripts have Probe 5b confidence but not Hindi GT-likelihood
        # difficulty. Rank instrument by Probe 5b mean_confidence *inverted*
        # (low confidence = hard) — labelled exploratory, not primary.
        conf = instrument_confidence_by_id(repo_root, condition=script)
        xs, ys, n_ok = [], [], 0
        for row in sarvam_rows:
            if row.get("script") != script or row.get("error"):
                continue
            iid = str(row["image_id"])
            if iid not in conf:
                continue
            n_ok += 1
            xs.append(-conf[iid])
            ys.append(finite_cer(row.get("sarvam_text") or "", row.get("ground_truth") or ""))
        res = spearman_perm_p(xs, ys)
        lines.append(
            format_result_block(
                f"Exploratory, underpowered: {script} plains "
                f"(instrument = −Probe5b mean_confidence, not GT log p)",
                res,
            )
        )

    lines.append(
        "Local engines on the full 60-id Hindi pool (independent of Sarvam n):\n"
    )
    for engine in ("tesseract", "surya"):
        eng = pair_engine_cohort(repo_root, engine, hindi_ids, variant="plain")
        res = spearman_perm_p(
            [r["instrument_difficulty"] for r in eng],
            [r["engine_cer"] for r in eng],
        )
        lines.append(
            format_result_block(
                f"Full Hindi-60 pool, {engine} CER (no extra API)", res
            )
        )

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5b rank-correlation (Decision #89). "
            "Does not call the Sarvam API. Refuses to compute ρ unless "
            "--compute-now is set, so a default invocation cannot peek."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/stage5b_rank_correlation.md"),
    )
    parser.add_argument(
        "--compute-now",
        action="store_true",
        help="Write ρ / permutation p / n. Only after the user confirms the page set.",
    )
    args = parser.parse_args(argv)
    write_stage5b_report(args.repo_root, args.out, compute=args.compute_now)
    print(f"Wrote {args.out} (compute={args.compute_now})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

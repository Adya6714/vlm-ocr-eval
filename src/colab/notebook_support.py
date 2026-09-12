"""
Shared report writers and checkpoint discovery for notebooks/colab_run.ipynb.

Why a module, not only notebook cells: the notebook must stay thin enough
to read, and these functions can be syntax-checked on a laptop without a
T4. They do not load models.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.analyze_pos0_null import fmt, load_jsonl, summarize

REQUIRED_CKPT = (
    "checkpoint_hindi_natural_seed0.pt",
    "checkpoint_hindi_natural_seed1.pt",
    "checkpoint_hindi_natural_seed2.pt",
    "tokenizer_hindi_natural.json",
)

DRIVE_CANDIDATES = (
    Path("/content/drive/MyDrive/vlm-ocr-eval/checkpoints"),
    Path("/content/drive/MyDrive/vlm-ocr-eval/checkpoint"),
    Path("/content/drive/MyDrive/checkpoints"),
    Path("/content/drive/MyDrive/vlm-ocr-eval"),
)

POS_BUCKETS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-9", 2, 9),
    ("10-19", 10, 19),
    ("20-39", 20, 39),
    ("40+", 40, 10**9),
)


def find_checkpoint_dir(extra: Iterable[Path] | None = None) -> Path:
    """
    Locate the four Hindi natural files. Print every path checked.

    Raises FileNotFoundError listing every candidate if none is complete.
    """
    checked: list[str] = []
    candidates = list(DRIVE_CANDIDATES)
    if extra:
        candidates = list(extra) + candidates
    for d in candidates:
        checked.append(str(d))
        print(f"[checkpoints] checking {d} exists={d.is_dir()}")
        if not d.is_dir():
            continue
        missing = [n for n in REQUIRED_CKPT if not (d / n).is_file()]
        if missing:
            print(f"[checkpoints] incomplete at {d}: missing {missing}")
            continue
        print(f"[checkpoints] FOUND complete set at {d}")
        return d
    raise FileNotFoundError(
        "Hindi instrument checkpoints not found. Checked, in order:\n  "
        + "\n  ".join(checked)
        + "\nNeed all of: "
        + ", ".join(REQUIRED_CKPT)
        + "\nMount Drive (Cell 2) and put them under "
        "MyDrive/vlm-ocr-eval/checkpoints/ (the path recorded in committed "
        "probe jsonl)."
    )


def copy_checkpoints(src: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_CKPT:
        target = dest / name
        shutil.copy2(src / name, target)
        print(f"[checkpoints] copied {name} -> {target} ({target.stat().st_size} bytes)")
    return dest


def write_tier0a(repo: Path) -> str:
    lines = [
        "# Tier 0a — E1 position-0 null control",
        "",
        "Producer: `src/probes/probe_pos0_null.py` (Colab T4 notebook).",
        "",
    ]
    all_rows = []
    missing = []
    for s in range(3):
        path = repo / f"data/probe_results/probe_pos0_null_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            missing.append(str(path))
            continue
        all_rows.extend(rows)
        lines.append(f"- **Seed {s}:** {fmt(summarize(rows))}")
    if missing:
        lines += ["", "**Incomplete.** Missing jsonl:"] + [f"- `{m}`" for m in missing]
    elif all_rows:
        lines.append(f"- **Pooled:** {fmt(summarize(all_rows))}")
        lines.append("")
        lines.append(
            "Headline is GT rank (median, IQR, fraction > 100), not 1/|V|."
        )
    text = "\n".join(lines) + "\n"
    out = repo / "docs" / "tier0a_pos0_null.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[tier0a] wrote {out}")
    return text


def _bucket_means(rows: list[dict]) -> dict[str, float | None]:
    acc: dict[str, list[float]] = {n: [] for n, _, _ in POS_BUCKETS}
    for row in rows:
        steps = row.get("step_log_p_gt") or []
        for i, lp in enumerate(steps):
            if lp is None or (isinstance(lp, float) and math.isnan(lp)):
                continue
            for name, lo, hi in POS_BUCKETS:
                if lo <= i <= hi:
                    acc[name].append(float(lp))
                    break
    return {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}


def write_tier0b(repo: Path) -> str:
    lines = [
        "# Tier 0b — derangement teacher-forcing",
        "",
        "Producer: `src/probes/probe_gt_mismatch.py` with `--derange-seed $s`.",
        "",
        "| seed | n | mean log p(GT) | median |",
        "|---|---:|---:|---:|",
    ]
    all_rows: list[dict] = []
    missing = []
    for s in range(3):
        path = repo / f"data/probe_results/probe_gt_mismatch_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            missing.append(str(path))
            continue
        all_rows.extend(rows)
        lps = [r["mean_log_p_gt"] for r in rows if r.get("mean_log_p_gt") is not None]
        import numpy as np

        lines.append(
            f"| {s} | {len(rows)} | {float(np.mean(lps)) if lps else 'n/a'} | "
            f"{float(np.median(lps)) if lps else 'n/a'} |"
        )
    lines += ["", "Per-bucket mean log p(GT) (pooled):", ""]
    if all_rows:
        bm = _bucket_means(all_rows)
        lines.append("| bucket | mean log p(GT) |")
        lines.append("|---|---:|")
        for name, _, _ in POS_BUCKETS:
            v = bm[name]
            lines.append(f"| {name} | {v if v is None else f'{v:.4f}'} |")
    if missing:
        lines += ["", "**Incomplete.** Missing:"] + [f"- `{m}`" for m in missing]
    text = "\n".join(lines) + "\n"
    out = repo / "docs" / "tier0b_gt_mismatch.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[tier0b] wrote {out}")
    return text


def write_tier0c(repo: Path) -> str:
    lines = [
        "# Tier 0c — cross-attention contribution norms",
        "",
        "Producer: `src/probes/probe_cross_attn_norms.py`.",
        "",
    ]
    missing = []
    for s in range(3):
        path = repo / f"data/probe_results/probe_cross_attn_norms_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            missing.append(str(path))
            continue
        lines.append(f"## Seed {s} (n={len(rows)})")
        by = {}
        for r in rows:
            by.setdefault(r.get("condition"), []).append(r)
        for cond, recs in sorted(by.items()):
            # per_layer is on each row
            n_layers = 0
            if recs and recs[0].get("per_layer"):
                n_layers = len(recs[0]["per_layer"])
            means = []
            for ℓ in range(n_layers):
                vals = []
                for r in recs:
                    pl = r.get("per_layer") or []
                    if ℓ < len(pl) and pl[ℓ].get("mean_ratio") is not None:
                        vals.append(float(pl[ℓ]["mean_ratio"]))
                means.append(sum(vals) / len(vals) if vals else None)
            pretty = " ".join(
                f"L{i}={m:.4f}" if m is not None else f"L{i}=n/a"
                for i, m in enumerate(means)
            )
            lines.append(f"- **{cond}:** {pretty}")
        lines.append("")
    if missing:
        lines += ["**Incomplete.** Missing:"] + [f"- `{m}`" for m in missing]
    text = "\n".join(lines) + "\n"
    out = repo / "docs" / "tier0c_cross_attn_norms.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[tier0c] wrote {out}")
    return text


def write_tier0d(repo: Path) -> str:
    lines = [
        "# Tier 0d — noise and scrambled teacher-forcing",
        "",
        "Sibling jsonl (`probe_gt_likelihood_extra_*`) so committed real/blank files stay intact.",
        "",
        "| seed | condition | n | mean log p(GT) | mean entropy |",
        "|---|---|---:|---:|---:|",
    ]
    import numpy as np

    missing = []
    for s in range(3):
        path = repo / f"data/probe_results/probe_gt_likelihood_extra_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            missing.append(str(path))
            continue
        by = {}
        for r in rows:
            by.setdefault(r.get("condition"), []).append(r)
        for cond, recs in sorted(by.items()):
            lps = [r["mean_log_p_gt"] for r in recs if r.get("mean_log_p_gt") is not None]
            ents = [r["mean_entropy"] for r in recs if r.get("mean_entropy") is not None]
            lines.append(
                f"| {s} | {cond} | {len(recs)} | "
                f"{float(np.mean(lps)) if lps else 'n/a'} | "
                f"{float(np.mean(ents)) if ents else 'n/a'} |"
            )
    if missing:
        lines += ["", "**Incomplete.** Missing:"] + [f"- `{m}`" for m in missing]
    text = "\n".join(lines) + "\n"
    out = repo / "docs" / "tier0d_noise_scrambled.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[tier0d] wrote {out}")
    return text


def choose_demo_backbone(vram: dict) -> dict:
    """
    Close Decision #3 from measured T4 peaks.

    Prefer any model with peak < 14 GB (2 GB reserved for later modules).
    Among those, pick the smallest peak. If none fit, say so — do not pick.
    """
    scored = []
    for rec in vram.values() if isinstance(vram, dict) else []:
        if not isinstance(rec, dict):
            continue
        peak = rec.get("peak_gb")
        if not isinstance(peak, (int, float)):
            continue
        scored.append(rec)
    if not scored:
        n = len(vram) if isinstance(vram, dict) else 0
        return {
            "selected": None,
            "reason": (
                f"no candidate could be measured — all {n} VRAM dummy runs "
                "crashed before producing a peak figure; see stderr above "
                "for the actual errors."
            ),
            "candidates": [],
        }
    fits = [r for r in scored if r.get("fits_t4_14gb_headroom")]
    if not fits:
        peaks = ", ".join(
            f"{r.get('model_id', '?')}={r['peak_gb']:.2f}GB" for r in scored
        )
        return {
            "selected": None,
            "reason": (
                "measured but exceeded 14 GB peak (2 GB headroom reserved): "
                + peaks
            ),
            "candidates": scored,
        }
    best = min(fits, key=lambda r: r["peak_gb"])
    return {
        "selected": best["model_id"],
        "reason": (
            f"lowest peak among models with ≥2 GB T4 headroom "
            f"({best['peak_gb']:.2f} GB peak, {best.get('headroom_gb', 16-best['peak_gb']):.2f} GB left)"
        ),
        "candidates": scored,
    }


def append_decision_3_close(repo: Path, choice: dict) -> None:
    path = repo / "DECISIONS.md"
    text = path.read_text(encoding="utf-8")
    if "### 83." in text or "Close Decision #3 from T4 LoRA VRAM" in text:
        print("[decision] #83 already present; not duplicating")
        return
    selected = choice.get("selected")
    block = f"""

---

### 83. Close Decision #3 from T4 LoRA VRAM

**Decision:** demo backbone = `{selected}`.

**Why:** {choice.get("reason")}

Measured peaks (this Colab T4 run) are in `docs/demo_lora_vram.json`
and `docs/demo_lora_inspect.json`. #3's original "not yet benchmarked"
sentence is historical; this entry is the close.

**Date:** 2026-09-12
"""
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")
    print(f"[decision] appended #83 selected={selected}")

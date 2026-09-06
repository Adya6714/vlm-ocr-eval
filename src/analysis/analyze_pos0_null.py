"""
Pool E1 jsonl into docs/pos0_null_control.md.

Not run until probe_pos0_null.py has written three seed files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def geom_mean(xs: list[float]) -> float:
    xs = np.clip(np.asarray(xs, dtype=np.float64), 1e-30, 1.0)
    return float(np.exp(np.mean(np.log(xs))))


def summarize(rows: list[dict]) -> dict:
    ranks = [r["gt_rank"] for r in rows]
    return {
        "n": len(rows),
        "geom_p_non_argmax": geom_mean([r["geom_p_non_argmax_100"] for r in rows]),
        "geom_p_gt_derange": geom_mean([r["p_gt_derange"] for r in rows]),
        "rank_median": float(np.median(ranks)),
        "rank_q25": float(np.percentile(ranks, 25)),
        "rank_q75": float(np.percentile(ranks, 75)),
        "frac_rank_gt_100": float(np.mean([r["gt_rank_gt_100"] for r in rows])),
        "mean_entropy": float(np.mean([r["entropy_nats"] for r in rows])),
        "geom_p_gt": geom_mean([r["p_gt"] for r in rows]),
    }


def fmt(d: dict) -> str:
    return (
        f"n={d['n']}; rank median {d['rank_median']:.1f} "
        f"IQR [{d['rank_q25']:.1f}, {d['rank_q75']:.1f}]; "
        f"P(rank>100)={d['frac_rank_gt_100']:.3f}; "
        f"H̄={d['mean_entropy']:.3f} nats; "
        f"geom p(non-argmax)={d['geom_p_non_argmax']:.3e}; "
        f"geom p(GT derange)={d['geom_p_gt_derange']:.3e}; "
        f"geom p(GT matched)={d['geom_p_gt']:.3e}"
    )


def main() -> None:
    lines = [
        "# E1 position-0 null control",
        "",
        "Producer: `src/probes/probe_pos0_null.py`.",
        "",
    ]
    all_rows = []
    missing = []
    for s in range(3):
        path = _ROOT / f"data/probe_results/probe_pos0_null_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            missing.append(str(path))
            continue
        all_rows.extend(rows)
        lines.append(f"- **Seed {s}:** {fmt(summarize(rows))}")
    if missing:
        lines.append("")
        lines.append("**Not computed.** Missing jsonl:")
        for m in missing:
            lines.append(f"- `{m}`")
        lines.append("")
        lines.append("Cause: `checkpoint_hindi_natural_seed{0,1,2}.pt` not in this checkout.")
    elif all_rows:
        lines.append(f"- **Pooled:** {fmt(summarize(all_rows))}")
        lines.append("")
        lines.append(
            "Headline is GT rank (median, IQR, fraction > 100), not a uniform 1/|V| comparison."
        )
    out = _ROOT / "docs" / "pos0_null_control.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

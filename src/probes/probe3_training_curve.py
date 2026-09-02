"""
Probe 3b — training-curve disambiguation for the blank-control finding.

Probe 3 shows ~0.99 mean confidence on real, blank, AND noise images
with ~0.10 accuracy. Two explanations are observationally identical on
the final checkpoint alone:
  (a) confidence is structurally ungrounded in the image (language-prior
      guessing regardless of input), or
  (b) the model is simply undertrained at 19.5M params / 5000 steps.

This probe separates them by re-running Probe 3's real-vs-blank
confidence comparison at multiple training steps. Requires intermediate
weight snapshots from train.py --keep-snapshots (the default resume
checkpoint overwrites every checkpoint_every steps).

Interpretation printed in the output JSON:
  - loss falls sharply while real-minus-blank gap stays ~0 → (a)
  - gap opens as training progresses → (b)

Called from: Colab after a training run with --keep-snapshots, or
locally once snapshots exist under --output-root.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from probe3_blank_control import make_blank
from probe_utils import (
    load_model_from_checkpoint_file,
    load_tokenizer,
    resolve_checkpoint_for_step,
    run_generate,
)

_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "eval")
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from equivalence_tables import tier1_equivalent
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP


def is_correct(prediction: str, ground_truth: str, script: str) -> bool:
    """
    Same Tier 1/2 scoring as probe5_calibration.py so accuracy is
    comparable across probes on real manifest lines.
    """
    if tier1_equivalent(prediction, ground_truth):
        return True
    if script in SCRIPT_MAP:
        return tier2_equivalent(ground_truth, prediction, script)
    return False


def parse_steps(steps_arg: str) -> list[int]:
    """Comma-separated step list, e.g. '200,500,1000,2000,5000'."""
    steps = [int(s.strip()) for s in steps_arg.split(",") if s.strip()]
    if not steps:
        raise ValueError("--steps must list at least one training step")
    return sorted(set(steps))


def evaluate_at_step(
    manifest_rows: list[dict],
    tokenizer,
    output_root: Path,
    script: str,
    condition: str,
    seed: int,
    step: int,
    device: torch.device,
) -> dict:
    """
    Run Probe 3's real-vs-blank comparison on one checkpoint snapshot.

    Returns per-step aggregates: training loss stored in the checkpoint,
    mean confidence on real and blank crops, their gap, and Tier 1/2
    accuracy on real images only.
    """
    ckpt_path = resolve_checkpoint_for_step(output_root, script, condition, seed, step)
    model, ckpt = load_model_from_checkpoint_file(
        ckpt_path, tokenizer, script, condition, device,
    )

    real_confs, blank_confs, correct_flags = [], [], []
    for i, row in enumerate(manifest_rows):
        real_img = Image.open(row["image_path"])
        blank_img = make_blank(real_img)

        real_out = run_generate(model, tokenizer, real_img, device)
        blank_out = run_generate(model, tokenizer, blank_img, device)

        real_conf = float(np.mean(real_out["step_confidences"])) if real_out["step_confidences"] else 0.0
        blank_conf = float(np.mean(blank_out["step_confidences"])) if blank_out["step_confidences"] else 0.0
        real_confs.append(real_conf)
        blank_confs.append(blank_conf)
        correct_flags.append(is_correct(real_out["text"], row["text"], script))

        if i % 10 == 0:
            print(f"[probe3_curve step={step}] {i + 1}/{len(manifest_rows)} images done")

    mean_real = float(np.mean(real_confs))
    mean_blank = float(np.mean(blank_confs))
    accuracy = float(np.mean(correct_flags))
    training_loss = ckpt.get("loss")

    return {
        "step": step,
        "checkpoint": str(ckpt_path),
        "training_loss": training_loss,
        "mean_confidence_real": mean_real,
        "mean_confidence_blank": mean_blank,
        "real_minus_blank_gap": mean_real - mean_blank,
        "accuracy": accuracy,
        "n_samples": len(manifest_rows),
    }


def interpret_curve(points: list[dict]) -> str:
    """
    State which competing explanation the step series supports.

    Uses the first and last available points: a two-order-of-magnitude
    loss drop with a flat real-blank gap supports (a); a growing gap
    supports (b). Mixed or inconclusive cases are stated plainly.
    """
    if len(points) < 2:
        return (
            "Insufficient steps for training-curve interpretation — "
            "need at least two checkpoints."
        )

    first, last = points[0], points[-1]
    loss_first = first.get("training_loss")
    loss_last = last.get("training_loss")
    gap_first = first["real_minus_blank_gap"]
    gap_last = last["real_minus_blank_gap"]

    loss_ratio = None
    if loss_first is not None and loss_last is not None and loss_last > 0:
        loss_ratio = loss_first / loss_last

    gap_flat = abs(gap_last - gap_first) < 0.05
    gap_opened = gap_last - gap_first > 0.05

    if loss_ratio is not None and loss_ratio >= 100 and gap_flat:
        return (
            f"Loss fell ~{loss_ratio:.0f}× (from {loss_first:.4f} to {loss_last:.4f}) "
            f"while the real-minus-blank confidence gap stayed near "
            f"{gap_last:+.3f} — strong evidence for (a): the model learned "
            f"the language prior and did not learn to use the image."
        )
    if gap_opened:
        return (
            f"The real-minus-blank confidence gap widened from {gap_first:+.3f} "
            f"to {gap_last:+.3f} over training — supports (b): more training "
            f"may teach the model to condition on the image, not just the prior."
        )
    if loss_ratio is not None and loss_ratio < 10:
        return (
            f"Loss changed only modestly ({loss_first:.4f} → {loss_last:.4f}) — "
            f"training may not have progressed enough to distinguish (a) from (b)."
        )
    return (
        f"Loss moved ({loss_first} → {loss_last}) and gap moved "
        f"({gap_first:+.3f} → {gap_last:+.3f}); neither (a) nor (b) is "
        f"decisive without more steps or a larger sample."
    )


def write_plot(points: list[dict], out_path: Path, title: str) -> str | None:
    """
    Optional PNG beside the JSON report. Skipped quietly if matplotlib is
    not installed — the JSON + printed table remain the source of truth.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe3_curve] matplotlib not installed — skipping plot")
        return None

    steps = [p["step"] for p in points]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    losses = [p.get("training_loss") for p in points]
    if any(v is not None for v in losses):
        axes[0, 0].plot(steps, losses, marker="o")
        axes[0, 0].set_ylabel("training loss")
        axes[0, 0].set_yscale("log")

    axes[0, 1].plot(steps, [p["mean_confidence_real"] for p in points], marker="o", label="real")
    axes[0, 1].plot(steps, [p["mean_confidence_blank"] for p in points], marker="s", label="blank")
    axes[0, 1].set_ylabel("mean confidence")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].plot(steps, [p["real_minus_blank_gap"] for p in points], marker="o", color="C2")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 0].set_ylabel("real − blank gap")

    axes[1, 1].plot(steps, [p["accuracy"] for p in points], marker="o", color="C3")
    axes[1, 1].set_ylabel("accuracy (real images)")
    axes[1, 1].set_ylim(0, 1.05)

    for ax in axes[1, :]:
        ax.set_xlabel("training step")

    plot_path = out_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def print_summary_table(points: list[dict]) -> None:
    """Human-readable table for Colab logs before the JSON is downloaded."""
    header = (
        f"{'step':>6}  {'loss':>10}  {'conf_real':>10}  {'conf_blank':>10}  "
        f"{'gap':>8}  {'acc':>8}"
    )
    print(header)
    print("-" * len(header))
    for p in points:
        loss = p.get("training_loss")
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        print(
            f"{p['step']:>6}  {loss_s:>10}  {p['mean_confidence_real']:>10.4f}  "
            f"{p['mean_confidence_blank']:>10.4f}  {p['real_minus_blank_gap']:>+8.4f}  "
            f"{p['accuracy']:>8.3f}"
        )


def run_probe3_training_curve(
    manifest_path: Path,
    output_root: Path,
    script: str,
    condition: str,
    seed: int,
    steps: list[int],
    out_path: Path,
    n_samples: int = 30,
    device_str: str = "cpu",
) -> dict:
    device = torch.device(device_str)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    sample_rows = random.Random(0).sample(rows, min(n_samples, len(rows)))

    print(
        f"[probe3_curve] {script} {condition} seed={seed}: "
        f"{len(steps)} steps × {len(sample_rows)} images"
    )

    tokenizer = load_tokenizer(output_root, script, condition)

    points = []
    for step in steps:
        print(f"\n[probe3_curve] === evaluating step {step} ===")
        points.append(
            evaluate_at_step(
                sample_rows, tokenizer, output_root, script, condition, seed, step, device,
            )
        )

    interpretation = interpret_curve(points)
    print_summary_table(points)
    print(f"\n[probe3_curve] interpretation:\n  {interpretation}")

    report = {
        "script": script,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "n_samples": len(sample_rows),
        "points": points,
        "interpretation": interpretation,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[probe3_curve] wrote {out_path}")

    title = f"Probe 3 training curve — {script} {condition} seed{seed}"
    plot_path = write_plot(points, out_path, title)
    if plot_path:
        report["plot_path"] = plot_path
        print(f"[probe3_curve] wrote {plot_path}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe 3 training curve: real-vs-blank confidence across training steps",
    )
    ap.add_argument("--manifest", required=True, help="Real line-crop manifest JSONL")
    ap.add_argument("--output-root", required=True, help="Directory with step snapshots")
    ap.add_argument("--script", required=True, choices=["hindi", "bengali"])
    ap.add_argument("--condition", required=True, choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--steps", required=True,
        help="Comma-separated training steps, e.g. 200,500,1000,2000,5000",
    )
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_probe3_training_curve(
        Path(args.manifest),
        Path(args.output_root),
        args.script,
        args.condition,
        args.seed,
        parse_steps(args.steps),
        Path(args.out),
        args.n_samples,
        args.device,
    )


if __name__ == "__main__":
    main()

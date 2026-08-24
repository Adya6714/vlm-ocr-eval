"""
Probe 1 orchestrator: runs the instrument model's training 9 times --
3 glyph-frequency conditions (natural / flattened / inverted) x 3 seeds
each -- per DECISIONS.md #14. Three seeds per condition is
non-negotiable: with one seed, the whole observed exposure-vs-complexity
spread could just be seed noise.

This sits ABOVE train.py's own resumability (which resumes a single
run mid-training if a Colab session dies). This file adds a second,
run-level layer: if the orchestrator itself is restarted, it should
skip any of the 9 runs that already finished, not just resume a
run that was interrupted mid-way. train.py's checkpoint step count is
the source of truth for "did this run finish" -- no separate state is
kept here, so there's only one place that answers "is run X done."

Does NOT yet include the glyph-level fixed-effects analysis
(IMPLEMENTATION.md's "fit per-glyph-cluster accuracy against log
exposure with glyph fixed effects") -- that requires trained model
output that doesn't exist until these 9 runs actually complete. That
analysis is the next piece to build once real training data and a
real Stage 1 renderer output exist; this file is scoped to getting the
9 runs themselves running and organized.

INTERFACE DEPENDENCY (same one train.py has): expects one manifest
file per condition (natural/flattened/inverted), each in the
{"image_path", "text"} JSONL shape train.py's LineDataset expects.
Manifest paths are passed in explicitly, not guessed from a Stage 1
naming convention that hasn't been confirmed yet.
"""

import argparse
import json
import os
import sys

import torch

# probe1_exposure.py lives in src/probes/, but train.py lives in
# src/models/instrument/ -- rather than keeping a second copy of this
# file inside that directory (which was tried first and rejected: two
# copies of one file WILL drift out of sync eventually, and this repo
# has already hit enough bugs from manual file duplication/retyping
# this session alone), compute the relative path to that directory and
# add it to sys.path so the import below works regardless of the
# current working directory this script is launched from.
_INSTRUMENT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "instrument")
)
sys.path.insert(0, _INSTRUMENT_DIR)

from train import train as run_training, checkpoint_path

CONDITIONS = ["natural", "flattened", "inverted"]
SEEDS = [0, 1, 2]


class Args:
    """
    Plain container matching train.py's argparse.Namespace shape, so
    run_training() (train.py's train() function) can be called directly
    in-process rather than shelling out per run. In-process is
    preferred here over subprocess: 9 sequential subprocess launches
    would each pay Python/torch import startup cost again, which adds
    up meaningfully across 9 runs on a Colab session with a limited
    time budget.
    """

    def __init__(self, manifest, condition, seed, output_root,
                 batch_size, lr, total_steps, log_every, checkpoint_every):
        self.manifest = manifest
        self.condition = condition
        self.seed = seed
        self.output_root = output_root
        self.batch_size = batch_size
        self.lr = lr
        self.total_steps = total_steps
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every


def run_is_complete(output_root: str, condition: str, seed: int, total_steps: int) -> bool:
    """
    A run counts as complete if its checkpoint exists AND has already
    reached total_steps -- not just "a checkpoint file exists," since
    that could be a partially-finished run from an earlier, smaller
    --total-steps value. Reading the checkpoint's stored step count
    directly, rather than trusting a separate "done" marker file, keeps
    a single source of truth (the checkpoint itself) instead of two
    things that could drift out of sync.
    """
    path = checkpoint_path(output_root, condition, seed)
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["step"] >= total_steps


def run_probe1(manifests: dict, output_root: str, total_steps: int,
                batch_size: int = 32, lr: float = 3e-4,
                log_every: int = 50, checkpoint_every: int = 200) -> None:
    """
    Runs all 9 (condition, seed) combinations in sequence, skipping any
    already complete. manifests: {"natural": path, "flattened": path,
    "inverted": path} -- one manifest per condition, built by Stage 1's
    renderer at each glyph-frequency setting.
    """
    plan = [(c, s) for c in CONDITIONS for s in SEEDS]
    print(f"Probe 1: {len(plan)} runs planned ({len(CONDITIONS)} conditions x {len(SEEDS)} seeds)")

    for condition, seed in plan:
        if run_is_complete(output_root, condition, seed, total_steps):
            print(f"[{condition} seed={seed}] already complete, skipping")
            continue

        if condition not in manifests:
            raise ValueError(
                f"No manifest provided for condition '{condition}'. "
                f"Probe 1 needs all three: {CONDITIONS}. Got: {list(manifests.keys())}"
            )

        print(f"\n{'='*60}\n[{condition} seed={seed}] starting\n{'='*60}")
        args = Args(
            manifest=manifests[condition],
            condition=condition,
            seed=seed,
            output_root=output_root,
            batch_size=batch_size,
            lr=lr,
            total_steps=total_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
        )
        run_training(args)

    print("\nProbe 1: all 9 runs complete.")
    print("Next step (not yet built): glyph-level fixed-effects analysis "
          "over these 9 checkpoints' per-glyph-cluster accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-manifest", required=True)
    parser.add_argument("--flattened-manifest", required=True)
    parser.add_argument("--inverted-manifest", required=True)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    manifests = {
        "natural": args.natural_manifest,
        "flattened": args.flattened_manifest,
        "inverted": args.inverted_manifest,
    }
    run_probe1(
        manifests, args.output_root, args.total_steps,
        args.batch_size, args.lr, args.log_every, args.checkpoint_every,
    )

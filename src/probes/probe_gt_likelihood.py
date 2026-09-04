"""
Teacher-forced ground-truth likelihood + predictive entropy.

Why this exists: mean max-softmax over *self-generated* tokens is
upward-biased by construction (the decoder scores the token it just
chose). Length-normalized log p of those same tokens is still the same
channel. This probe scores the model on the *ground-truth* token
sequence under teacher forcing, and records full-distribution entropy
at each step — estimators that do not inherit the argmax self-selection
bias.

Conditions (same Hindi Tier C pool as Probe 5b / Probe 6 real_plain —
those two image sets are identical: all 60 GT rows under Random(0)
with n_samples ≥ 60):
  - real: the plain scan, resized to canonical height
  - blank: solid white of the same size (probe3 make_blank)

Per record: per-step log p(gt_token), per-step entropy H(p), then
length-normalized means. Resumable append+skip on (condition, image_path).

Do not confuse with Probe 5 calibration confidence — that is still
mean max-softmax on greedy self-generation. This file is the paper's
estimator-independence check.

Called from: Colab after hindi/natural checkpoints exist. Not run in
the laptop checkout by default (GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe3_blank_control import make_blank  # noqa: E402
from probe5b_zeroshot_floor import (  # noqa: E402
    IN_DISTRIBUTION_LANGUAGE,
    resolve_repo_root,
)
from probe_attention_ablation import build_hindi_sample  # noqa: E402
from probe_utils import (  # noqa: E402
    load_model_and_tokenizer,
    prepare_image_tensor,
    resize_to_canonical_height,
)

_INSTRUMENT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "instrument")
)
if _INSTRUMENT_DIR not in sys.path:
    sys.path.insert(0, _INSTRUMENT_DIR)

from generate import generate  # noqa: E402
from train import checkpoint_path  # noqa: E402

CONDITIONS = ("real", "blank")


def gt_force_ids(tokenizer, ground_truth: str) -> list[int]:
    """
    Token ids AFTER <BOS> for teacher forcing, including trailing <EOS>.

    Matches generate.force_next_ids convention and train-time encode()
    (BOS + content + EOS). Unknown graphemes map to <RARE> — still a
    valid forced id, and the log-prob of <RARE> is an honest score for
    OOV clusters.
    """
    ids = tokenizer.encode(ground_truth or "", add_special_tokens=True)
    if not ids or ids[0] != tokenizer.cluster_to_id["<BOS>"]:
        raise ValueError("encode() must prepend <BOS>")
    return ids[1:]


def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """H(p) = −∑ p log p over the full vocabulary (nats)."""
    p = probs.clamp_min(eps)
    p = p / p.sum()
    return float(-(p * torch.log(p)).sum().item())


def load_completed_keys(out_path: Path) -> set[tuple[str, str]]:
    """Resume set: (condition, image_path) already written."""
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["condition"], row["image_path"]))
    return done


def build_tasks(data_root: Path, repo_root: Path, n_samples: int) -> list[dict]:
    """
    Real + blank tasks on the Probe 5b Hindi sample.

    Same Random(0) draw as probe5b / attention ablation / probe6's
    full Hindi pool (intersection verified: 60/60 identical paths).
    """
    hindi = build_hindi_sample(data_root, repo_root, n_samples)
    tasks: list[dict] = []
    for item in hindi:
        tasks.append({
            "condition": "real",
            "row": item["row"],
            "image_path": item["image_path"],
            "source_path": item["image_path"],
        })
        tasks.append({
            "condition": "blank",
            "row": item["row"],
            "image_path": item["image_path"],
            "source_path": item["image_path"],
            "blank_source_path": item["image_path"],
        })
    return tasks


def score_one(
    model,
    tokenizer,
    image: Image.Image,
    ground_truth: str,
    device: torch.device,
) -> dict:
    """
    Teacher-force the GT sequence on one image; return likelihood stats.

    step_confidences from generate() under force_next_ids are already
    p(forced_token), not max-softmax — that is the GT likelihood at
    each step. Entropy needs return_full_probs.
    """
    forced = gt_force_ids(tokenizer, ground_truth)
    if not forced:
        return {
            "n_gt_tokens": 0,
            "mean_log_p_gt": None,
            "mean_entropy": None,
            "sum_log_p_gt": None,
            "step_log_p_gt": [],
            "step_entropy": [],
            "n_rare_forced": 0,
        }

    tensor = prepare_image_tensor(image).to(device)
    out = generate(
        model,
        tensor,
        tokenizer,
        device=device,
        force_next_ids=forced,
        max_len=len(forced),
        return_full_probs=True,
    )

    rare_id = tokenizer.cluster_to_id["<RARE>"]
    step_log_p: list[float] = []
    step_entropy: list[float] = []
    n_rare = 0
    for i, (p_gt, probs) in enumerate(zip(out["step_confidences"], out["step_probs"])):
        # p_gt is already probs[forced_id] from generate()
        p = max(float(p_gt), 1e-12)
        step_log_p.append(math.log(p))
        step_entropy.append(shannon_entropy(probs))
        if forced[i] == rare_id:
            n_rare += 1

    return {
        "n_gt_tokens": len(step_log_p),
        "mean_log_p_gt": float(np.mean(step_log_p)) if step_log_p else None,
        "mean_entropy": float(np.mean(step_entropy)) if step_entropy else None,
        "sum_log_p_gt": float(np.sum(step_log_p)) if step_log_p else None,
        "step_log_p_gt": step_log_p,
        "step_entropy": step_entropy,
        "n_rare_forced": n_rare,
        # Echo: under teacher forcing these are p(gt), not argmax mass.
        "step_p_gt": list(out["step_confidences"]),
    }


def print_dry_run(
    tasks: list[dict],
    output_root: Path,
    script: str,
    condition: str,
    seed: int,
    out_path: Path,
) -> None:
    """List planned work without loading weights or touching the GPU."""
    ckpt = checkpoint_path(str(output_root), script, condition, seed)
    completed = load_completed_keys(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]
    by_cond = {c: sum(1 for t in tasks if t["condition"] == c) for c in CONDITIONS}
    print(f"[gt_likelihood] DRY-RUN seed={seed}")
    print(f"  checkpoint: {ckpt}  exists={Path(ckpt).exists()}")
    print(f"  out: {out_path}")
    print(f"  tasks total={len(tasks)}  by_condition={by_cond}")
    print(f"  already_done={len(completed)}  pending={len(pending)}")
    for t in pending[:3]:
        print(
            f"  sample pending: condition={t['condition']} "
            f"id={t['row'].get('id')} path={t['image_path']}"
        )
    if len(pending) > 3:
        print(f"  ... and {len(pending) - 3} more")


def run_probe(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str = "cpu",
    dry_run: bool = False,
) -> None:
    """
    Teacher-forced GT likelihood for one hindi/natural checkpoint.

    Checkpoint/resume: append+skip by (condition, image_path). Progress
    printed every image — silence would look like a hang on Colab.
    """
    repo_root = resolve_repo_root(data_root)
    tasks = build_tasks(data_root, repo_root, n_samples)

    if dry_run:
        print_dry_run(tasks, output_root, script, condition, seed, out_path)
        return

    device = torch.device(device_str)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)
    print(f"[gt_likelihood] checkpoint: {ckpt}")
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"missing checkpoint at {ckpt} (DECISIONS.md #47 script-scoped name)"
        )

    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device,
    )

    completed = load_completed_keys(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]
    total = len(tasks)
    already = total - len(pending)
    print(
        f"[gt_likelihood] {script}/{condition}/seed={seed}: "
        f"{total} tasks ({already} done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            row = task["row"]
            gt_text = row.get("text") or ""
            if task["condition"] == "blank":
                base = resize_to_canonical_height(Image.open(task["blank_source_path"]))
                image = make_blank(base)
            else:
                image = resize_to_canonical_height(Image.open(task["source_path"]))

            body = score_one(model, tokenizer, image, gt_text, device)
            record = {
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "condition": task["condition"],
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": gt_text,
                "estimator": "teacher_forced_gt_loglik_and_entropy",
                "checkpoint_path": ckpt,
                **body,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            done = already + i + 1
            mean_lp = body["mean_log_p_gt"]
            mean_h = body["mean_entropy"]
            lp_str = f"{mean_lp:.4f}" if mean_lp is not None else "n/a"
            h_str = f"{mean_h:.4f}" if mean_h is not None else "n/a"
            print(
                f"[gt_likelihood] {done}/{total} "
                f"cond={task['condition']} id={row.get('id')} "
                f"n_tok={body['n_gt_tokens']} mean_log_p={lp_str} "
                f"mean_H={h_str}"
            )

    # End-of-run condition means from the full file (includes resumed rows).
    by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("condition") in by_cond:
            by_cond[rec["condition"]].append(rec)
    print(f"[gt_likelihood] wrote {out_path}")
    for cond in CONDITIONS:
        recs = by_cond[cond]
        lps = [r["mean_log_p_gt"] for r in recs if r.get("mean_log_p_gt") is not None]
        ents = [r["mean_entropy"] for r in recs if r.get("mean_entropy") is not None]
        print(
            f"  {cond}: n={len(recs)} "
            f"mean_log_p_gt={float(np.mean(lps)) if lps else 'n/a'} "
            f"mean_entropy={float(np.mean(ents)) if ents else 'n/a'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Teacher-forced GT log-likelihood + predictive entropy "
            "(estimator-independence check for Claim B)"
        ),
    )
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition",
        default="natural",
        choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True, help="Checkpoint directory")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
        help="Data root with raw/ ground truth (default: OCR_DATA_ROOT or data/)",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help=(
            "Cap on Hindi images; same Random(0) draw as Probe 5b "
            "(pool is currently 60, so default 100 → full pool)"
        ),
    )
    ap.add_argument("--out", required=True, help="Output jsonl path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks/checkpoint paths; do not load model or write",
    )
    args = ap.parse_args()

    run_probe(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

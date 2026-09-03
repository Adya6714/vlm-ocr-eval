"""
Attention / encoder-memory ablation — Claim B mechanism probe.

Claim B (correlational so far): mean confidence barely changes across
real Hindi, blank, and never-seen-script images (Probe 3 / 5b). This
probe asks the mechanistic follow-up: does the decoder's next-token
distribution depend on the encoder's output *at all*, or is confidence
almost entirely the autoregressive prior?

Method (inference only, existing hindi/natural checkpoints):
  1. generate() with full encoder memory.
  2. generate() again with encoder memory replaced by zeros *before*
     memory_projection (prior-only).
  3. For per-step KL / top-1 / prior-sufficiency, re-score the
     zero-memory decoder under the *full-memory token prefixes*
     (teacher forcing) so sequence divergence does not confound the
     distribution comparison. Independent zero-memory greedy still
     supplies mean_confidence_zero for the headline contrast.
  4. Sample set = Probe 5b's Hindi condition (same Random(0) draw,
     same n_samples default), seeds 0/1/2, so numbers sit on the same
     statistical footing.

Outputs: data/probe_results/attention_ablation_hindi_natural_seed{N}.jsonl
Analysis: src/analysis/analyze_attention_ablation.py →
docs/attention_ablation_analysis.md (DECISIONS.md #56).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# probe_utils inserts the instrument dir; generate helpers live there.
_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe5b_zeroshot_floor import (  # noqa: E402
    IN_DISTRIBUTION_LANGUAGE,
    load_ground_truth_rows,
    resolve_image_path,
    resolve_repo_root,
)
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

from generate import generate, kl_divergence, prior_sufficiency  # noqa: E402


def build_hindi_sample(
    data_root: Path,
    repo_root: Path,
    n_samples: int,
) -> list[dict]:
    """
    Hindi image list matching Probe 5b's in-distribution condition.

    Uses the same Random(0) draw as probe5b_zeroshot_floor.build_task_list
    so attention-ablation results are directly comparable to the Probe 5b
    Hindi confidence numbers. When n_samples exceeds the Tier C Hindi
    pool (currently 60), both probes take the full pool — that is the
    "same n as probe5b" guarantee, not a hard 100.
    """
    import random

    hindi_rows = load_ground_truth_rows(data_root, IN_DISTRIBUTION_LANGUAGE)
    rng = random.Random(0)
    hindi_sample = rng.sample(hindi_rows, min(n_samples, len(hindi_rows)))
    tasks = []
    for row in hindi_sample:
        tasks.append({
            "language": IN_DISTRIBUTION_LANGUAGE,
            "row": row,
            "image_path": str(resolve_image_path(row, repo_root)),
        })
    return tasks


def load_completed_paths(out_path: Path) -> set[str]:
    """Resume set: image_path keys already written to the jsonl."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        done.add(json.loads(line)["image_path"])
    return done


def compare_distributions(
    probs_full: list[torch.Tensor],
    probs_zero: list[torch.Tensor],
    tokenizer,
) -> list[dict]:
    """
    Per-step KL(full || zero), top-1 agreement, and prior sufficiency
    under a shared prefix (equal-length probability lists).

    Primary KL direction is KL(p_full || p_zero) — how surprising the
    image-conditioned distribution is under the prior. Also records
    KL(p_zero || p_full) because it is cheap.
    """
    n = min(len(probs_full), len(probs_zero))
    steps = []
    for i in range(n):
        p = probs_full[i]
        q = probs_zero[i]
        top_full = int(torch.argmax(p).item())
        top_zero = int(torch.argmax(q).item())
        steps.append({
            "step": i,
            "kl_full_given_zero": kl_divergence(p, q),
            "kl_zero_given_full": kl_divergence(q, p),
            "top1_agree": top_full == top_zero,
            "prior_sufficiency": prior_sufficiency(p, q),
            "conf_full": float(p.max().item()),
            "conf_zero": float(q.max().item()),
            "token_full": tokenizer.id_to_cluster.get(top_full, "<RARE>"),
            "token_zero": tokenizer.id_to_cluster.get(top_zero, "<RARE>"),
        })
    return steps


def ablate_one_image(model, tokenizer, image: Image.Image, device: torch.device) -> dict:
    """
    Full-memory greedy + zero-memory greedy + shared-prefix zero re-score.

    Returns the jsonl record body for one image (caller adds metadata).
    Full softmax vectors are used in-memory for KL then discarded so
    the committed probe_results stay small (DECISIONS.md #56).
    """
    tensor = prepare_image_tensor(image).to(device)

    full = generate(
        model, tensor, tokenizer, device=device, return_full_probs=True,
    )
    zero_indep = generate(
        model, tensor, tokenizer, device=device, zero_encoder_memory=True,
    )
    # Re-score zero-memory under the full-memory token path (after BOS).
    forced = full["token_ids"][1:]
    zero_tf = generate(
        model,
        tensor,
        tokenizer,
        device=device,
        zero_encoder_memory=True,
        return_full_probs=True,
        force_next_ids=forced,
        max_len=len(forced),
    )

    step_metrics = compare_distributions(
        full["step_probs"], zero_tf["step_probs"], tokenizer,
    )
    mean_conf_full = (
        float(np.mean(full["step_confidences"])) if full["step_confidences"] else None
    )
    mean_conf_zero = (
        float(np.mean(zero_indep["step_confidences"]))
        if zero_indep["step_confidences"]
        else None
    )
    kls = [s["kl_full_given_zero"] for s in step_metrics]
    agrees = [s["top1_agree"] for s in step_metrics]
    prior_suff = [s["prior_sufficiency"] for s in step_metrics]

    return {
        "text_full": full["text"],
        "text_zero": zero_indep["text"],
        "token_ids_full": full["token_ids"],
        "token_ids_zero": zero_indep["token_ids"],
        "mean_confidence_full": mean_conf_full,
        "mean_confidence_zero": mean_conf_zero,
        "confidence_delta": (
            mean_conf_full - mean_conf_zero
            if mean_conf_full is not None and mean_conf_zero is not None
            else None
        ),
        "step_confidences_full": full["step_confidences"],
        "step_confidences_zero": zero_indep["step_confidences"],
        "step_top_k_full": full["step_top_k"],
        "step_top_k_zero": zero_indep["step_top_k"],
        "n_steps_compared": len(step_metrics),
        "mean_kl_full_given_zero": float(np.mean(kls)) if kls else None,
        "mean_kl_zero_given_full": (
            float(np.mean([s["kl_zero_given_full"] for s in step_metrics]))
            if step_metrics
            else None
        ),
        "top1_agreement_rate": float(np.mean(agrees)) if agrees else None,
        "mean_prior_sufficiency": float(np.mean(prior_suff)) if prior_suff else None,
        "step_metrics": step_metrics,
        # Document metric choices on every record so the jsonl is
        # self-describing without needing the analysis script open.
        "kl_direction": "KL(full || zero)",
        "prior_sufficiency_definition": "sum_i min(p_full[i], p_zero[i]) = 1 - TV",
        "prefix_alignment": "teacher_force_zero_on_full_tokens",
    }


def run_attention_ablation(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str = "cpu",
) -> None:
    """
    Run the ablation over the Probe 5b Hindi sample for one checkpoint.

    Checkpoint/resume: append+skip by image_path. Progress every image
    (per-image cost is two+ generates — silence looks like a hang).
    """
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device,
    )

    repo_root = resolve_repo_root(data_root)
    tasks = build_hindi_sample(data_root, repo_root, n_samples)
    completed = load_completed_paths(out_path)
    pending = [t for t in tasks if t["image_path"] not in completed]

    total = len(tasks)
    already = total - len(pending)
    print(
        f"[attention_ablation] {script}/{condition}/seed={seed}: "
        f"{total} images ({already} done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            row = task["row"]
            image = resize_to_canonical_height(Image.open(task["image_path"]))
            body = ablate_one_image(model, tokenizer, image, device)
            record = {
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": row.get("text"),
                **body,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            done_count = already + i + 1
            delta = record.get("confidence_delta")
            delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
            print(
                f"[attention_ablation] {done_count}/{total} "
                f"id={row.get('id')}  "
                f"conf_full={record['mean_confidence_full']:.4f}  "
                f"conf_zero={record['mean_confidence_zero']:.4f}  "
                f"Δ={delta_str}  "
                f"mean_KL={record['mean_kl_full_given_zero']:.4f}  "
                f"top1_agree={record['top1_agreement_rate']:.3f}  "
                f"prior_suff={record['mean_prior_sufficiency']:.3f}"
            )

    print(f"[attention_ablation] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encoder-memory ablation: does confidence depend on the image?",
    )
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition", default="natural", choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True, help="Directory with trained checkpoints")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
        help="Data root containing raw/ ground truth (default: OCR_DATA_ROOT or data/)",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Cap on Hindi images; same Random(0) draw as Probe 5b (pool may be smaller)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_attention_ablation(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
    )


if __name__ == "__main__":
    main()

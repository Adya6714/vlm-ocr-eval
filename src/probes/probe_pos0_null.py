"""
E1: position-0 null controls on the same 60 Hindi images × 3 seeds.

Why this exists: the paper declines a uniform-over-V comparison for
position 0. Reviewers still need (c) GT rank in the first-token
softmax, plus (a) mass on a random non-argmax symbol, (b) p(GT) under
a derangement of image–reference pairs, and (d) Shannon entropy.

All four are computed from the full softmax at the first generate()
step (after BOS), same 180 (image, reference) pairs as Probe 5b Hindi.

Prefix: free-run first step on the real image (not GT teacher forcing)
for (a)(c)(d). (b) is teacher-forced only in the sense that the scored
symbol is another image's GT first grapheme, still at position 0 of a
fresh decoder given that image.

Requires checkpoint_hindi_natural_seed{N}.pt. Inference-only. CPU ok.

Output: data/probe_results/probe_pos0_null_hindi_natural_seed{N}.jsonl
Analysis: src/analysis/analyze_pos0_null.py → docs/pos0_null_control.md
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

_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe_attention_ablation import build_hindi_sample  # noqa: E402
from probe5b_zeroshot_floor import resolve_repo_root  # noqa: E402
from probe_gt_likelihood import gt_force_ids, shannon_entropy  # noqa: E402
from probe_gt_mismatch import derangement_indices  # noqa: E402
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

N_NON_ARGMAX_DRAWS = 100
RANK_TAIL = 100


def load_completed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["image_path"])
    return done


def pos0_softmax(model, tokenizer, image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Full vocab softmax at the first generated position (after BOS).

    One generate() step with max_len=1 would still run the encoder;
    we request return_full_probs and take step 0.
    """
    tensor = prepare_image_tensor(image).to(device)
    out = generate(
        model,
        tensor,
        tokenizer,
        device=device,
        return_full_probs=True,
        max_len=1,
    )
    return out["step_probs"][0]


def gt_first_id(tokenizer, ground_truth: str) -> int:
    """First content token id after BOS (not EOS)."""
    forced = gt_force_ids(tokenizer, ground_truth)
    if not forced:
        raise ValueError("empty GT force ids")
    return int(forced[0])


def analyze_probs(
    probs: torch.Tensor,
    gt_id: int,
    rng: np.random.Generator,
) -> dict:
    """(a)(c)(d) on one position-0 distribution."""
    p = probs.detach().cpu().double()
    p = p / p.sum()
    vocab = p.numel()
    argmax = int(torch.argmax(p).item())
    # Rank: 1 = highest probability. Ties: average rank of the tied set.
    order = torch.argsort(p, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, vocab + 1, dtype=torch.float64)
    gt_rank = float(ranks[gt_id].item())
    entropy = shannon_entropy(p.float())
    p_gt = float(p[gt_id].item())
    others = [i for i in range(vocab) if i != argmax]
    draws = rng.choice(others, size=N_NON_ARGMAX_DRAWS, replace=True)
    p_draw = np.array([float(p[int(i)].item()) for i in draws], dtype=np.float64)
    p_draw = np.clip(p_draw, 1e-30, 1.0)
    geom_non_argmax = float(np.exp(np.mean(np.log(p_draw))))
    return {
        "argmax_id": argmax,
        "gt_id": gt_id,
        "p_gt": p_gt,
        "gt_rank": gt_rank,
        "gt_rank_gt_100": gt_rank > RANK_TAIL,
        "entropy_nats": entropy,
        "geom_p_non_argmax_100": geom_non_argmax,
        "vocab_size": vocab,
    }


def run_seed(
    output_root: Path,
    data_root: Path,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str,
) -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(
        output_root, "hindi", "natural", seed, device
    )
    repo_root = resolve_repo_root(data_root)
    tasks = build_hindi_sample(data_root, repo_root, n_samples)
    n = len(tasks)
    rng_derange = np.random.default_rng(seed)
    perm = derangement_indices(n, rng_derange)
    done = load_completed(out_path)
    pending = [t for t in tasks if t["image_path"] not in done]
    print(
        f"[pos0_null] seed={seed} {n} images, {len(pending)} remaining, device={device}"
    )
    cache_probs: dict[str, torch.Tensor] = {}

    def softmax_for(task: dict) -> torch.Tensor:
        key = task["image_path"]
        if key not in cache_probs:
            image = resize_to_canonical_height(Image.open(task["image_path"]))
            cache_probs[key] = pos0_softmax(model, tokenizer, image, device)
        return cache_probs[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = n - len(pending)
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            idx = next(j for j, t in enumerate(tasks) if t["image_path"] == task["image_path"])
            partner = tasks[int(perm[idx])]
            rng_a = np.random.default_rng(seed * 1_000_003 + idx)
            p_self = softmax_for(task)
            gt_id = gt_first_id(tokenizer, task["row"]["text"])
            partner_gt_id = gt_first_id(tokenizer, partner["row"]["text"])
            stats_self = analyze_probs(p_self, gt_id, rng_a)
            p_this = p_self.detach().cpu().double()
            p_this = p_this / p_this.sum()
            # Derangement: score this image's position-0 distro at the
            # paired image's first GT token (image_i, reference_j).
            p_gt_derange = float(p_this[partner_gt_id].item())
            record = {
                "seed": seed,
                "image_path": task["image_path"],
                "image_id": task["row"].get("id"),
                "ground_truth": task["row"].get("text"),
                "paired_image_id": partner["row"].get("id"),
                "p_gt_derange": p_gt_derange,
                **stats_self,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[pos0_null] {already + i + 1}/{n} id={record['image_id']} "
                f"rank={record['gt_rank']:.0f} p_gt={record['p_gt']:.3e} "
                f"H={record['entropy_nats']:.3f}"
            )
    print(f"[pos0_null] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 position-0 null controls")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--data-root", default=os.environ.get("OCR_DATA_ROOT", "data"))
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_seed(
        Path(args.output_root),
        Path(args.data_root),
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
    )


if __name__ == "__main__":
    main()

"""
Teacher-forced full-softmax vs 5-gram prior (Step 4b).

Why a separate probe: probe_gt_likelihood already runs return_full_probs
but only keeps entropy. Dumping every softmax would explode jsonl size.
This probe computes KL(model || 5-gram) and argmax agreement **online**
and stores two scalars per step.

Same 60-image Hindi pool as probe_gt_likelihood (build_hindi_sample /
Random(0)). Real images only unless --extra-conditions is passed.

Needs the instrument checkpoint + tokenizer. Resume on
(condition, image_path). Progress per image.

Called from Colab or CPU: see docs/ngram_kl_argmax.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROBES_DIR = Path(__file__).resolve().parent
_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
_INSTRUMENT = Path(__file__).resolve().parents[1] / "models" / "instrument"
for p in (_PROBES_DIR, _ANALYSIS, _INSTRUMENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate import generate, kl_divergence  # noqa: E402
from ngram_kl_argmax import (  # noqa: E402
    context_at_step,
    ngram_next_dist,
    scatter_onto_tokenizer,
)
from position_matched_ngrams import (  # noqa: E402
    ALPHA,
    EOS,
    build_ngram_counts,
    graphemes,
    train_eval_split,
)
from probe_gt_likelihood import (  # noqa: E402
    gt_force_ids,
    load_completed_keys,
    render_condition_image,
    shannon_entropy,
)
from probe5b_zeroshot_floor import resolve_repo_root  # noqa: E402
from probe_attention_ablation import build_hindi_sample  # noqa: E402
from probe_utils import load_model_and_tokenizer, prepare_image_tensor  # noqa: E402
from train import checkpoint_path  # noqa: E402


def build_real_tasks(data_root: Path, repo_root: Path, n_samples: int, conditions: tuple[str, ...]):
    hindi = build_hindi_sample(data_root, repo_root, n_samples)
    tasks = []
    for item in hindi:
        for cond in conditions:
            tasks.append(
                {
                    "condition": cond,
                    "row": item["row"],
                    "image_path": item["image_path"],
                    "source_path": item["image_path"],
                }
            )
    return tasks


def load_5gram(repo: Path, tokenizer):
    """
    5-gram on the 4a split, alphabet = train graphemes ∪ {EOS}.

    Tokenizer mapping happens at scatter time so OOV graphemes hit <RARE>
    instead of pretending the 5-gram lives in BPE space.
    """
    train_texts, _eval_texts, v_est = train_eval_split(repo)
    counts, context_counts = build_ngram_counts(train_texts, n=5)
    alphabet: set[str] = set()
    for t in train_texts:
        alphabet.update(graphemes(t))
    alphabet.add(EOS)
    cluster_to_id = tokenizer.cluster_to_id
    id_to_cluster = tokenizer.id_to_cluster
    vocab_size = len(tokenizer)
    rare_id = cluster_to_id["<RARE>"]
    return {
        "counts": counts,
        "context_counts": context_counts,
        "alphabet": alphabet,
        "v_est": v_est,
        "cluster_to_id": cluster_to_id,
        "id_to_cluster": id_to_cluster,
        "vocab_size": vocab_size,
        "rare_id": rare_id,
    }


def score_one_ngram(model, tokenizer, image, ground_truth, device, gram) -> dict:
    """
    Teacher-force GT; at each step compare full softmax to the 5-gram.

    Why scatter then KL: the decoder is defined on tokenizer ids, the
    5-gram on graphemes. KL is only defined after both are distributions
    on the same V. Argmax agreement uses those same vectors (model
    argmax vs 5-gram argmax after scatter).
    """
    forced = gt_force_ids(tokenizer, ground_truth)
    if not forced:
        return {
            "n_gt_tokens": 0,
            "step_kl_m_5gram": [],
            "step_argmax_agree": [],
            "step_entropy": [],
            "step_p_gt": [],
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

    kls: list[float] = []
    agrees: list[bool] = []
    ents: list[float] = []
    for i, probs in enumerate(out["step_probs"]):
        p = probs.detach().cpu().float().numpy()
        if p.ndim > 1:
            p = p.reshape(-1)
        ctx = context_at_step(ground_truth, i, n=5)
        dist = ngram_next_dist(
            ctx,
            gram["counts"],
            gram["context_counts"],
            gram["alphabet"],
            gram["v_est"],
            ALPHA,
        )
        q = scatter_onto_tokenizer(
            dist,
            gram["cluster_to_id"],
            gram["vocab_size"],
            gram["rare_id"],
        )
        kls.append(kl_np_torch(p, q))
        agrees.append(int(p.argmax()) == int(q.argmax()))
        ents.append(shannon_entropy(probs))

    return {
        "n_gt_tokens": len(forced),
        "step_kl_m_5gram": kls,
        "step_argmax_agree": agrees,
        "step_entropy": ents,
        "step_p_gt": list(out["step_confidences"]),
        "estimator": "teacher_forced_kl_model_||_5gram",
    }


def kl_np_torch(p: np.ndarray, q: np.ndarray) -> float:
    return kl_divergence(torch.from_numpy(p), torch.from_numpy(q))


def run_probe(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str,
    dry_run: bool,
    image_conditions: tuple[str, ...],
) -> None:
    repo_root = resolve_repo_root(data_root)
    tasks = build_real_tasks(data_root, repo_root, n_samples, image_conditions)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)

    if dry_run:
        print(f"[ngram_kl] DRY-RUN seed={seed} ckpt={ckpt} exists={Path(ckpt).exists()}")
        print(f"  out={out_path} n_tasks={len(tasks)} conditions={image_conditions}")
        return

    if not Path(ckpt).exists():
        raise FileNotFoundError(f"missing checkpoint at {ckpt}")

    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device,
    )
    gram = load_5gram(repo_root, tokenizer)
    print(
        f"[ngram_kl] 5-gram v_est={gram['v_est']} |alphabet|={len(gram['alphabet'])} "
        f"tokenizer V={gram['vocab_size']}"
    )

    completed = load_completed_keys(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]
    total = len(tasks)
    already = total - len(pending)
    print(
        f"[ngram_kl] {script}/{condition}/seed={seed}: "
        f"{total} tasks ({already} done, {len(pending)} remaining)"
    )

    rng = np.random.default_rng(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            row = task["row"]
            gt = row.get("text") or ""
            image = render_condition_image(task, rng)
            body = score_one_ngram(model, tokenizer, image, gt, device, gram)
            rec = {
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "condition": task["condition"],
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": gt,
                "checkpoint_path": ckpt,
                **body,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mean_kl = float(np.mean(body["step_kl_m_5gram"])) if body["step_kl_m_5gram"] else float("nan")
            agree = float(np.mean(body["step_argmax_agree"])) if body["step_argmax_agree"] else float("nan")
            print(
                f"[ngram_kl] {already + i + 1}/{total} "
                f"cond={task['condition']} id={row.get('id')} "
                f"n_tok={body['n_gt_tokens']} mean_KL={mean_kl:.4f} "
                f"argmax_agree={agree:.3f}"
            )
    print(f"[ngram_kl] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4b: KL(model || 5-gram) + argmax agreement")
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument("--condition", default="natural", choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--data-root", default=os.environ.get("OCR_DATA_ROOT", "data"))
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--extra-conditions",
        nargs="*",
        default=[],
        choices=["blank", "noise", "scrambled"],
        help="Default is real only (Table 6 exclusivity on text-bearing input).",
    )
    args = ap.parse_args()
    conds = ("real",) + tuple(args.extra_conditions)
    seen = set()
    conds = tuple(c for c in conds if not (c in seen or seen.add(c)))
    run_probe(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
        args.dry_run,
        conds,
    )


if __name__ == "__main__":
    main()

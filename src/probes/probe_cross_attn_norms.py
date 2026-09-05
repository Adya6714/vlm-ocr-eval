"""
Cross-attention contribution norm per decoder layer.

Why this exists: encoder-zeroing shows the *distribution* can move while
max-softmax stays high, but that still does not quantify how large the
cross-attn pathway is in the residual stream on a normal forward pass.
For each decoder layer we record

    r_ℓ = mean_t  ‖cross-attn output_t‖₂  /  ‖residual stream_t‖₂

averaged over teacher-forced GT positions, for real vs blank images.
If r_ℓ is tiny on both, the encoder pathway is carrying little residual
mass — a direct "modality bypass" measurement (paper item 8 / missing
from attention_ablation_analysis.md).

Implementation note: InstrumentDecoder uses PyTorch
TransformerDecoderLayer with norm_first=True. We wrap each layer's
forward so we see the residual *before* the cross-attn residual add and
the cross-attn block output that gets added.

Forward passes only; no weight updates. Checkpoint/resume + per-image
progress.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe3_blank_control import make_blank  # noqa: E402
from probe_gt_likelihood import gt_force_ids  # noqa: E402
from probe5b_zeroshot_floor import resolve_repo_root  # noqa: E402
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

from train import checkpoint_path  # noqa: E402

CONDITIONS = ("real", "blank")


def install_cross_attn_hooks(decoder: nn.Module) -> tuple[list, list[dict]]:
    """
    Wrap each TransformerDecoderLayer.forward to log contribution ratios.

    Returns (handles_as_restore_fns, stores) where stores[ℓ] accumulates
    per-forward lists of per-position ratios for layer ℓ.
    """
    stores: list[dict] = []
    restores: list = []
    layers = list(decoder.layers.layers)

    for layer_idx, layer in enumerate(layers):
        store = {"ratios": [], "contrib_norms": [], "stream_norms": []}
        stores.append(store)
        orig_forward = layer.forward

        def make_forward(orig, st, lyr):
            def wrapped(
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=False,
                memory_is_causal=False,
            ):
                # Mirror torch.nn.TransformerDecoderLayer with norm_first.
                x = tgt
                if lyr.norm_first:
                    x = x + lyr._sa_block(
                        lyr.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
                    )
                    stream = x
                    contrib = lyr._mha_block(
                        lyr.norm2(x),
                        memory,
                        memory_mask,
                        memory_key_padding_mask,
                        memory_is_causal,
                    )
                    # [B, T, D] → per-position L2
                    c_n = contrib.detach().float().norm(dim=-1)  # [B, T]
                    s_n = stream.detach().float().norm(dim=-1).clamp_min(1e-8)
                    ratios = (c_n / s_n).reshape(-1)
                    st["ratios"].append(ratios.cpu())
                    st["contrib_norms"].append(c_n.reshape(-1).cpu())
                    st["stream_norms"].append(s_n.reshape(-1).cpu())
                    x = x + contrib
                    x = x + lyr._ff_block(lyr.norm3(x))
                else:
                    # Unexpected for this instrument model; still run original.
                    return orig(
                        tgt,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        tgt_is_causal=tgt_is_causal,
                        memory_is_causal=memory_is_causal,
                    )
                return x

            return wrapped

        layer.forward = make_forward(orig_forward, store, layer)

        def make_restore(lyr, orig):
            def restore():
                lyr.forward = orig

            return restore

        restores.append(make_restore(layer, orig_forward))

    return restores, stores


def clear_stores(stores: list[dict]) -> None:
    for st in stores:
        st["ratios"].clear()
        st["contrib_norms"].clear()
        st["stream_norms"].clear()


def summarize_stores(stores: list[dict]) -> list[dict]:
    out = []
    for ℓ, st in enumerate(stores):
        if not st["ratios"]:
            out.append({
                "layer": ℓ,
                "mean_ratio": None,
                "n_positions": 0,
            })
            continue
        ratios = torch.cat(st["ratios"]).numpy()
        out.append({
            "layer": ℓ,
            "mean_ratio": float(ratios.mean()),
            "median_ratio": float(np.median(ratios)),
            "mean_contrib_norm": float(torch.cat(st["contrib_norms"]).mean()),
            "mean_stream_norm": float(torch.cat(st["stream_norms"]).mean()),
            "n_positions": int(ratios.size),
        })
    return out


@torch.no_grad()
def score_cross_attn(
    model,
    tokenizer,
    image: Image.Image,
    ground_truth: str,
    device: torch.device,
    stores: list[dict],
) -> dict:
    """
    One teacher-forced full-sequence decoder pass with hooks live.

    Uses model.forward (encoder + memory_projection + decoder) so
    cross-attn sees real projected memory — not the step-by-step
    generate() loop (which would re-accumulate ratios every step).
    """
    clear_stores(stores)
    forced = gt_force_ids(tokenizer, ground_truth)
    if not forced:
        return {"n_gt_tokens": 0, "per_layer": summarize_stores(stores)}

    bos = tokenizer.cluster_to_id["<BOS>"]
    # Teacher-forcing input: BOS + all but last forced token; targets are forced.
    # For contribution norms we only need a decoder forward over the GT prefix;
    # use full BOS+content (excluding final EOS is fine — include all forced).
    target_ids = torch.tensor([[bos] + forced[:-1]], device=device, dtype=torch.long)
    tensor = prepare_image_tensor(image).to(device)
    _ = model(tensor, target_ids)
    return {
        "n_gt_tokens": len(forced),
        "per_layer": summarize_stores(stores),
    }


def load_completed(out_path: Path) -> set[tuple[str, str]]:
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
    repo_root = resolve_repo_root(data_root)
    tasks = build_tasks(data_root, repo_root, n_samples)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)

    if dry_run:
        done = load_completed(out_path)
        pending = [t for t in tasks if (t["condition"], t["image_path"]) not in done]
        print(f"[cross_attn] DRY-RUN seed={seed}")
        print(f"  checkpoint: {ckpt} exists={Path(ckpt).exists()}")
        print(f"  tasks={len(tasks)} pending={len(pending)}")
        return

    device = torch.device(device_str)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"missing checkpoint at {ckpt}")
    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device
    )
    model.eval()
    restores, stores = install_cross_attn_hooks(model.decoder)

    completed = load_completed(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]
    total = len(tasks)
    already = total - len(pending)
    print(
        f"[cross_attn] {script}/{condition}/seed={seed}: "
        f"{total} tasks ({already} done, {len(pending)} remaining); "
        f"n_layers={len(stores)}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for i, task in enumerate(pending):
                if task["condition"] == "blank":
                    base = resize_to_canonical_height(
                        Image.open(task["blank_source_path"])
                    )
                    image = make_blank(base)
                else:
                    image = resize_to_canonical_height(Image.open(task["source_path"]))
                gt = task["row"].get("text") or ""
                body = score_cross_attn(model, tokenizer, image, gt, device, stores)
                record = {
                    "checkpoint_script": script,
                    "training_condition": condition,
                    "seed": seed,
                    "condition": task["condition"],
                    "image_path": task["image_path"],
                    "image_id": task["row"].get("id"),
                    "ground_truth": gt,
                    "estimator": "cross_attn_contrib_over_residual",
                    "checkpoint_path": ckpt,
                    **body,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done = already + i + 1
                means = [
                    (p["layer"], p["mean_ratio"])
                    for p in body["per_layer"]
                    if p["mean_ratio"] is not None
                ]
                mean_str = " ".join(f"L{ℓ}={r:.4f}" for ℓ, r in means)
                print(
                    f"[cross_attn] {done}/{total} cond={task['condition']} "
                    f"id={task['row'].get('id')} {mean_str}"
                )
    finally:
        for restore in restores:
            restore()

    # End-of-run: mean ratio per layer × condition
    by: dict[str, list[list[float]]] = {c: [[] for _ in stores] for c in CONDITIONS}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("condition") not in by:
            continue
        for p in rec.get("per_layer") or []:
            if p.get("mean_ratio") is not None:
                by[rec["condition"]][p["layer"]].append(p["mean_ratio"])
    print(f"[cross_attn] wrote {out_path}")
    for cond in CONDITIONS:
        parts = []
        for ℓ, vals in enumerate(by[cond]):
            if vals:
                parts.append(f"L{ℓ}={float(np.mean(vals)):.4f}")
        print(f"  {cond}: n_images≈{len(by[cond][0]) if by[cond][0] else 0}  " + " ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition",
        default="natural",
        choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
    )
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
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
        args.dry_run,
    )


if __name__ == "__main__":
    main()

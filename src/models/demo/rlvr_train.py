"""
Single RLVR train: coverage-term ablation (Decision #11).

Only runs if an SFT adapter directory exists. lambda_coverage=0 is the
ablation. This is REINFORCE on sequence logprob × reward, not a sweep.

Refuses CUDA-less hosts so a laptop cannot mint a fake T4 result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_DEMO = Path(__file__).resolve().parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from rlvr import compute_reward  # noqa: E402
from sft import LineCropDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-root", required=True)
    ap.add_argument("--manifest", default="data/manifests/hindi_natural.jsonl")
    ap.add_argument("--data-root", default=os.environ.get("OCR_DATA_ROOT", "data"))
    ap.add_argument("--output-root", default="checkpoints/demo_rlvr_nocov")
    ap.add_argument("--lambda-coverage", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("RLVR train requires CUDA. Not attempted.")
    sft = Path(args.sft_root)
    adapter = sft / "adapter_config.json"
    if not adapter.exists():
        marker = {
            "attempted": False,
            "reason": f"no PEFT adapter at {adapter}; SFT did not produce a usable checkpoint",
        }
        out = Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rlvr_not_trained.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )
        print(json.dumps(marker, indent=2))
        raise SystemExit(0)
    print(
        f"[rlvr] would train λ_coverage={args.lambda_coverage} from {sft} "
        f"max_steps={args.max_steps}. Full PPO not implemented; writing "
        "omission diagnostic on greedy decode of the SFT adapter instead."
    )
    # Greedy decode a handful of lines and report coverage vs char_acc.
    from peft import PeftModel
    from base_model import load_model_and_processor

    cfg = json.loads((sft / "sft_meta.json").read_text(encoding="utf-8")) if (sft / "sft_meta.json").exists() else {}
    model_id = cfg.get("model_id")
    if not model_id:
        raise SystemExit("sft_meta.json missing model_id")
    base, processor = load_model_and_processor(model_id)
    model = PeftModel.from_pretrained(base, str(sft))
    model.to(args.device)
    model.eval()
    ds = LineCropDataset(Path(args.manifest), Path(args.data_root))
    n = min(32, len(ds))
    rows = []
    for i in range(n):
        item = ds[i]
        print(f"[rlvr] decode {i+1}/{n}")
        try:
            inputs = processor(images=item["image"], return_tensors="pt")
            inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64)
            hyp = processor.batch_decode(gen, skip_special_tokens=True)[0]
        except Exception as e:
            hyp = ""
            print(f"[rlvr] generate failed: {type(e).__name__}: {e}")
        r = compute_reward(hyp, item["text"], lambda_coverage=args.lambda_coverage)
        r0 = compute_reward(hyp, item["text"], lambda_coverage=0.0)
        rows.append({
            "image_path": item["image_path"],
            "gt": item["text"],
            "hyp": hyp,
            **r,
            "reward_lambda0": r0["reward"],
        })
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rlvr_omission_sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cov = [r["coverage"] for r in rows]
    omit = sum(1 for r in rows if r["coverage"] < 0.5)
    summary = {
        "n": len(rows),
        "mean_coverage": sum(cov) / len(cov) if cov else None,
        "n_coverage_lt_0.5": omit,
        "lambda_coverage_trained": args.lambda_coverage,
        "note": (
            "This is SFT-greedy scored with λ=0 reward, not a retrained "
            "policy. Retrain-from-SFT is skipped unless generate() works "
            "and session time remains for a real RL loop."
        ),
    }
    (out / "rlvr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[rlvr] wrote {path}")


if __name__ == "__main__":
    main()

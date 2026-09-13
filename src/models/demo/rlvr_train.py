"""
SFT-greedy coverage diagnostic (Decision #11 baseline), not a trainer.

Phase 2a: match SFT chat template, do not swallow generate errors,
score with emitted-only accuracy (Decision #88). No rollout/update.
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
from sft import (  # noqa: E402
    LineCropDataset,
    decode_continuation,
    encode_for_generate,
)


def load_completed_paths(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["image_path"])
    return done


def score_sft_greedy(
    sft_root: Path,
    manifest: Path,
    data_root: Path,
    output_root: Path,
    device_str: str,
    n_samples: int,
    lambda_coverage: float,
) -> dict:
    """
    Greedy decode n line crops with the SFT adapter; write jsonl + summary.

    Failures raise. Empty hyp is only a real empty continuation, not a
    caught exception. Resume: skip image_path already in the jsonl.
    """
    from peft import PeftModel
    from base_model import load_model_and_processor

    cfg = {}
    meta = sft_root / "sft_meta.json"
    if meta.exists():
        cfg = json.loads(meta.read_text(encoding="utf-8"))
    model_id = cfg.get("model_id")
    if not model_id:
        raise SystemExit("sft_meta.json missing model_id")

    device = torch.device(device_str)
    print(f"[rlvr] load base={model_id} adapter={sft_root} device={device}")
    base, processor = load_model_and_processor(model_id)
    model = PeftModel.from_pretrained(base, str(sft_root))
    model.to(device)
    model.eval()

    ds = LineCropDataset(manifest, data_root)
    n = min(n_samples, len(ds))
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "rlvr_omission_sample.jsonl"
    done = load_completed_paths(path)
    print(f"[rlvr] scoring {n} lines; {len(done)} already in {path}")

    with path.open("a", encoding="utf-8") as f:
        for i in range(n):
            item = ds[i]
            if item["image_path"] in done:
                print(f"[rlvr] skip {i+1}/{n} already scored")
                continue
            print(f"[rlvr] decode {i+1}/{n} {item['image_path']}")
            inputs = encode_for_generate(processor, item["image"], device_str)
            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64)
            hyp = decode_continuation(processor, gen, prompt_len)
            r = compute_reward(
                hyp,
                item["text"],
                lambda_coverage=lambda_coverage,
                use_emitted_only_acc=True,
            )
            r0 = compute_reward(
                hyp, item["text"], lambda_coverage=0.0, use_emitted_only_acc=True
            )
            row = {
                "image_path": item["image_path"],
                "gt": item["text"],
                "hyp": hyp,
                "hyp_empty": hyp == "",
                "prompt_len": prompt_len,
                **r,
                "reward_lambda0": r0["reward"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[rlvr] {i+1}/{n} coverage={r['coverage']:.3f} "
                f"emitted_acc={r['emitted_acc']:.3f} char_acc={r['char_acc']:.3f} "
                f"hyp_empty={hyp == ''} hyp_chars={len(hyp)}"
            )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Restrict summary to this run's first n dataset paths if file grew.
    want = {ds[i]["image_path"] for i in range(n)}
    rows = [r for r in rows if r["image_path"] in want]
    cov = [r["coverage"] for r in rows]
    n_empty = sum(1 for r in rows if r.get("hyp_empty") or r.get("hyp") == "")
    omit = sum(1 for r in rows if r["coverage"] < 0.5)
    summary = {
        "n": len(rows),
        "n_empty_hyp": n_empty,
        "mean_coverage": (sum(cov) / len(cov)) if cov else None,
        "n_coverage_lt_0.5": omit,
        "lambda_coverage_scored": lambda_coverage,
        "acc_used": "emitted_only",
        "chat_template": "sft_user_turn + add_generation_prompt=True",
        "swallowed_exceptions": False,
        "note": (
            "SFT-greedy scored with SFT chat template; not a retrained "
            "policy. No generate() exceptions were converted to hyp=''."
        ),
    }
    (output_root / "rlvr_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"[rlvr] wrote {path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-root", required=True)
    ap.add_argument("--manifest", default="data/manifests/hindi_natural.jsonl")
    ap.add_argument("--data-root", default=os.environ.get("OCR_DATA_ROOT", "data"))
    ap.add_argument("--output-root", default="checkpoints/demo_rlvr_sft_baseline")
    ap.add_argument("--lambda-coverage", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=30, help="Ignored; no trainer.")
    ap.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Line crops to greedy-decode (same default as the old diagnostic).",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.max_steps:
        print(
            f"[rlvr] --max-steps={args.max_steps} is not used "
            "(no policy update in this file)."
        )
    sft = Path(args.sft_root)
    adapter = sft / "adapter_config.json"
    if not adapter.exists():
        marker = {
            "attempted": False,
            "mean_coverage": None,
            "reason": f"no PEFT adapter at {adapter}; cannot re-baseline",
        }
        out = Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rlvr_not_trained.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )
        print(json.dumps(marker, indent=2))
        raise SystemExit(0)
    score_sft_greedy(
        sft,
        Path(args.manifest),
        Path(args.data_root),
        Path(args.output_root),
        args.device,
        args.n_samples,
        args.lambda_coverage,
    )


if __name__ == "__main__":
    main()

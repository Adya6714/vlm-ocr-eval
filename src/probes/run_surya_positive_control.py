"""
Tier 1 Surya positive-control attempts (a) old package (b) explicit GQA
reshape (c) stop. Called from notebooks/colab_run.ipynb. No
ignore_mismatched_sizes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path


PRE_LLAMA_PIN = "0.14.6"  # last 0.14.x on PyPI before later VLM work; 0.22 is llama.cpp
HF_ID = "vikp/surya_rec"


def pip_show_surya() -> str:
    p = subprocess.run(
        [sys.executable, "-m", "pip", "show", "surya-ocr"],
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    print(out or "[surya] pip show: package not installed")
    return out


def version_from_show(text: str) -> str | None:
    for line in text.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def llama_cpp_runtime() -> bool:
    try:
        import surya.inference.backends.llamacpp  # noqa: F401

        return True
    except Exception:
        return False


def try_old_surya_import() -> tuple[bool, str]:
    """
    Path (a): pin pre-0.22 and look for the torch recognition model.
    """
    notes = []
    cmd = [sys.executable, "-m", "pip", "install", "-q", f"surya-ocr=={PRE_LLAMA_PIN}"]
    print("[surya] (a) pip install", " ".join(cmd[3:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    notes.append((r.stdout or "")[-500:] + (r.stderr or "")[-500:])
    if r.returncode != 0:
        return False, f"pip install {PRE_LLAMA_PIN} failed\n" + "\n".join(notes)
    # Common historic import paths
    for path in (
        "surya.model.recognition.model",
        "surya.recognition.model",
        "surya.model.recognition.encoderdecoder",
    ):
        try:
            mod = __import__(path, fromlist=["*"])
            print(f"[surya] (a) imported {path}: {dir(mod)[:20]}")
            return True, f"imported {path}"
        except Exception as e:
            notes.append(f"{path}: {type(e).__name__}: {e}")
    return False, "no historic torch rec module\n" + "\n".join(notes)


def expand_gqa_key(key: str, tensor, q_out: int, kv_out: int):
    """
    Repeat KV heads so k_proj/v_proj out_features match q_proj.

    weight is [out, in] = [kv_out, hidden]. We reshape to
    [n_kv_heads, head_dim, hidden], repeat_interleave n_q/n_kv on heads,
    flatten to [q_out, hidden]. Same for bias [kv_out] → [q_out].

    This is an *approximation*: it copies shared KV heads; it is not the
    original GQA kernel. Documented because it is not ignore_mismatched_sizes.
    """
    import torch

    if tensor.ndim == 1:
        if tensor.shape[0] != kv_out:
            return tensor, False
        n_repeat = q_out // kv_out
        if n_repeat * kv_out != q_out:
            return tensor, False
        return tensor.repeat_interleave(n_repeat, dim=0), True
    if tensor.ndim != 2 or tensor.shape[0] != kv_out:
        return tensor, False
    n_repeat = q_out // kv_out
    if n_repeat * kv_out != q_out:
        return tensor, False
    head_dim = kv_out  # not necessarily; use ratio from q
    # Prefer: kv_out / n_kv == q_out / n_q. We only know sizes.
    # Treat kv_out rows as (n_kv * head) with head = kv_out // n_kv unknown.
    # Using repeat_interleave on the out axis in blocks of (kv_out // gcd).
    w = tensor.reshape(kv_out, -1)
    # Repeat each of kv_out rows n_repeat times would be wrong for GQA
    # (that duplicates entire rows, not heads). Block-repeat:
    n_kv = kv_out  # fallback
    # If q_out/kv_out is integer, group as n_kv= kv_out/head_dim... we don't
    # have head_dim. Standard Donut/MBart GQA: out_kv=256, out_q=1024, ratio=4.
    ratio = q_out // kv_out
    head_dim = 64  # common; verified against 256/4 and 1024/16
    n_kv = kv_out // head_dim
    n_q = q_out // head_dim
    if n_kv * head_dim != kv_out or n_q * head_dim != q_out:
        # fall back to naive row repeat (worse; still explicit)
        return w.repeat_interleave(ratio, dim=0), True
    w3 = w.view(n_kv, head_dim, w.shape[1])
    w3 = w3.repeat_interleave(n_q // n_kv, dim=0)
    return w3.reshape(q_out, w.shape[1]), True


def try_explicit_gqa_load(cache_dir: Path | None = None) -> tuple[bool, str]:
    """
    Path (b): load vikp/surya_rec, expand GQA k/v, drop MoE expert keys.
    """
    import torch
    from huggingface_hub import hf_hub_download

    notes = []
    try:
        cfg_path = hf_hub_download(HF_ID, "config.json")
        cfg = json.loads(Path(cfg_path).read_text())
        notes.append(f"architectures={cfg.get('architectures')}")
    except Exception as e:
        return False, f"could not fetch config: {e}"

    dropped = []
    remapped = []
    try:
        from transformers import AutoConfig

        # Do not use ignore_mismatched_sizes. Build state dict by hand.
        try:
            from transformers import VisionEncoderDecoderModel
        except Exception as e:
            return False, f"VisionEncoderDecoderModel import failed: {e}"

        # This path is expected to fail on GQA/MoE with a *documented*
        # reshape. We try to load safetensors, transform, then load_state_dict(strict=False)
        # only after dropping listed MoE keys — not a hidden library flag.
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download as dl

        st_path = dl(HF_ID, "model.safetensors")
        raw = load_file(st_path)
        new_sd = {}
        for k, v in raw.items():
            if ".moe.experts." in k:
                dropped.append(k)
                continue
            if k.endswith("k_proj.weight") or k.endswith("v_proj.weight") or k.endswith("k_proj.bias") or k.endswith("v_proj.bias"):
                if v.shape[0] == 256:
                    nv, ok = expand_gqa_key(k, v, q_out=1024, kv_out=256)
                    if ok:
                        remapped.append(f"{k} {tuple(v.shape)} -> {tuple(nv.shape)}")
                        new_sd[k] = nv
                        continue
            new_sd[k] = v
        notes.append(f"dropped_moe_n={len(dropped)}")
        notes.append("dropped_moe_examples=" + ", ".join(dropped[:8]))
        notes.append("gqa_remaps=" + "; ".join(remapped[:8]))
        notes.append(
            "MoE drop is NOT a defensible approximation of the trained "
            "layer-3 experts; it replaces an MoE FFN with missing fc1/fc2 "
            "and would randomly init those. Stopping before load_state_dict "
            "because that would not be the published vikp/surya_rec model."
        )
        return False, "\n".join(notes)
    except Exception:
        return False, traceback.format_exc()


def write_report(repo: Path, body: str) -> None:
    path = repo / "docs" / "tier1_surya_control.md"
    path.write_text(body, encoding="utf-8")
    print(body)
    print(f"[surya] wrote {path}")


def main(repo: Path) -> int:
    show = pip_show_surya()
    ver = version_from_show(show)
    lines = [
        "# Tier 1 — Surya positive control (Colab T4)",
        "",
        f"pip show version: `{ver}`",
        "",
        f"llama.cpp backend importable: **{llama_cpp_runtime()}**",
        "",
    ]
    if ver and ver.startswith("0.22"):
        lines.append(
            "This is the llama.cpp / SuryaInferenceManager line already "
            "documented as incompatible with full-softmax / encoder-zeroing."
        )
        lines.append("")
    ok_a, note_a = try_old_surya_import()
    lines += ["## Path (a) — pin surya-ocr==" + PRE_LLAMA_PIN, "", note_a, ""]
    ok_b, note_b = try_explicit_gqa_load()
    lines += ["## Path (b) — explicit GQA expand + listed MoE drop", "", note_b, ""]
    if not ok_a and not ok_b:
        lines += [
            "## Path (c) — not viable",
            "",
            "Surya is **not** a usable positive control for teacher-forced "
            "log p(GT), GT rank, or encoder-zeroing on this runtime.",
            "",
            "Do not use `ignore_mismatched_sizes`. Do not treat Table 1's "
            "46.8% exact-match as a position-0 softmax result.",
            "",
            "Working alternatives to try later (open-weight encoder-decoder "
            "OCR, torch, full logits): **PaddleOCR v4/v5 recognition head** "
            "(already in this repo as a string baseline, not a softmax "
            "control), **TrOCR / Donut** if a Devanagari checkpoint exists, "
            "or **granite-docling / SmolDocling** once Decision #3 is closed "
            "— those expose transformers logits. None of those is Table 1's "
            "Surya engine.",
            "",
        ]
        write_report(repo, "\n".join(lines) + "\n")
        return 1
    lines.append("A load path reported success; diagnostic loop was not auto-run from this helper — see notebook Cell 7 continuation.")
    write_report(repo, "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    raise SystemExit(main(repo))

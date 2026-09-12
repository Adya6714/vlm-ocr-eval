"""Generate notebooks/colab_run.ipynb. Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": [],
}


def md(src: str) -> None:
    NB["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in src.strip("\n").split("\n")]}
    )


def code(src: str) -> None:
    lines = src.strip("\n") + "\n"
    NB["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [ln + "\n" for ln in lines.split("\n")][:-1] + ([lines.split("\n")[-1] + "\n"] if lines.split("\n")[-1] else []),
            "execution_count": None,
            "outputs": [],
        }
    )
    # simpler:
    NB["cells"][-1]["source"] = [l + "\n" for l in src.strip("\n").split("\n")]


md("""# Colab T4 run — Tier 0a–d, Tier 1 Surya, Tier 2 Steps 1/2/4

**You** upload this notebook to Colab, set **Runtime → Change runtime type → T4 GPU**, and run cells **in order**.

This notebook does **not** edit `paper/main.tex`. It writes `docs/` and `data/probe_results/`.

The last two cells: first **print the git diff and stop**; only the cell after that commits and pushes, and only if you run it yourself.""")

md("## Cell 1 — environment check (must be a T4)")

code(r'''
import torch

print("=" * 72)
print("CELL 1 — GPU CHECK")
print("=" * 72)
ok = torch.cuda.is_available()
print(f"torch.cuda.is_available() = {ok}")
assert ok, (
    "NO CUDA. Runtime → Change runtime type → T4 GPU, then Runtime → Restart session, "
    "then re-run from Cell 1. Do not continue on CPU."
)
name = torch.cuda.get_device_name(0)
free_b, total_b = torch.cuda.mem_get_info(0)
free_gb, total_gb = free_b / 1e9, total_b / 1e9
print(f"device 0 name = {name!r}")
print(f"free VRAM = {free_gb:.2f} GB / {total_gb:.2f} GB")
print("=" * 72)
assert "T4" in name, (
    f"This notebook requires a Tesla T4. Got {name!r}. "
    "Runtime → Change runtime type → T4 GPU."
)
print("OK: T4 visible. Continue to Cell 2.")
''')

md("## Cell 2 — Drive mount (Colab UI OAuth only) + clone + checkpoints")

code(r'''
from google.colab import drive

print("=" * 72)
print("CELL 2 — DRIVE + CLONE")
print("=" * 72)
print("A Colab Drive popup should appear. Accept it. Do not use gcloud.")
drive.mount("/content/drive")

from pathlib import Path
import os, subprocess, sys

REPO_URL = "https://github.com/Adya6714/vlm-ocr-eval.git"
REPO = Path("/content/vlm-ocr-eval")
if (REPO / ".git").exists():
    print(f"[git] already cloned at {REPO}; fetching")
    subprocess.check_call(["git", "-C", str(REPO), "pull", "--ff-only"])
else:
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO)])

os.chdir(REPO)
print("[cwd]", os.getcwd())
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft", "regex", "scipy", "safetensors"])

os.environ["PYTHONPATH"] = str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from colab.notebook_support import copy_checkpoints, find_checkpoint_dir

ckpt_src = find_checkpoint_dir()
CKPT = copy_checkpoints(ckpt_src, REPO / "checkpoints")
os.environ["OCR_DATA_ROOT"] = str(REPO / "data")
os.environ["INSTRUMENT_CKPT"] = str(CKPT)
print("[ok] checkpoints ready at", CKPT)
print("Continue to Cell 3.")
''')

md("## Cell 3 — Tier 0a `probe_pos0_null.py`")

code(r'''
print("=" * 72)
print("CELL 3 — TIER 0a")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
CKPT = os.environ["INSTRUMENT_CKPT"]
failed = False
try:
    for s in (0, 1, 2):
        out = REPO / f"data/probe_results/probe_pos0_null_hindi_natural_seed{s}.jsonl"
        cmd = [
            sys.executable, "src/probes/probe_pos0_null.py",
            "--seed", str(s), "--device", "cuda",
            "--output-root", CKPT, "--data-root", "data", "--n-samples", "100",
            "--out", str(out),
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    sys.path.insert(0, str(REPO / "src"))
    from colab.notebook_support import write_tier0a
    write_tier0a(REPO)
except Exception:
    failed = True
    traceback.print_exc()
    print("CELL 3 FAILED. Later tier-0 cells will still run if you choose; this error is not silent.")
if failed:
    print("Stop or continue at your choice — Cell 4 is independent once checkpoints exist.")
else:
    print("Cell 3 done. Continue to Cell 4.")
''')

md("## Cell 4 — Tier 0b `probe_gt_mismatch.py`")

code(r'''
print("=" * 72)
print("CELL 4 — TIER 0b")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
CKPT = os.environ["INSTRUMENT_CKPT"]
try:
    for s in (0, 1, 2):
        out = REPO / f"data/probe_results/probe_gt_mismatch_hindi_natural_seed{s}.jsonl"
        cmd = [
            sys.executable, "src/probes/probe_gt_mismatch.py",
            "--script", "hindi", "--condition", "natural",
            "--seed", str(s), "--derange-seed", str(s),
            "--device", "cuda", "--output-root", CKPT, "--data-root", "data",
            "--n-samples", "100", "--out", str(out),
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    from colab.notebook_support import write_tier0b
    write_tier0b(REPO)
    print("Cell 4 done.")
except Exception:
    traceback.print_exc()
    print("CELL 4 FAILED — not skipping silently.")
''')

md("## Cell 5 — Tier 0c `probe_cross_attn_norms.py`")

code(r'''
print("=" * 72)
print("CELL 5 — TIER 0c")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
CKPT = os.environ["INSTRUMENT_CKPT"]
try:
    for s in (0, 1, 2):
        out = REPO / f"data/probe_results/probe_cross_attn_norms_hindi_natural_seed{s}.jsonl"
        cmd = [
            sys.executable, "src/probes/probe_cross_attn_norms.py",
            "--script", "hindi", "--condition", "natural",
            "--seed", str(s), "--device", "cuda",
            "--output-root", CKPT, "--data-root", "data",
            "--n-samples", "100", "--out", str(out),
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    from colab.notebook_support import write_tier0c
    write_tier0c(REPO)
    print("Cell 5 done.")
except Exception:
    traceback.print_exc()
    print("CELL 5 FAILED — not skipping silently.")
''')

md("## Cell 6 — Tier 0d noise + scrambled (`probe_gt_likelihood.py`)")

code(r'''
print("=" * 72)
print("CELL 6 — TIER 0d")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
CKPT = os.environ["INSTRUMENT_CKPT"]
try:
    for s in (0, 1, 2):
        out = REPO / f"data/probe_results/probe_gt_likelihood_extra_hindi_natural_seed{s}.jsonl"
        cmd = [
            sys.executable, "src/probes/probe_gt_likelihood.py",
            "--script", "hindi", "--condition", "natural",
            "--seed", str(s), "--device", "cuda",
            "--output-root", CKPT, "--data-root", "data",
            "--n-samples", "100",
            "--extra-conditions", "noise", "scrambled",
            "--out", str(out),
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    from colab.notebook_support import write_tier0d
    write_tier0d(REPO)
    print("Cell 6 done.")
except Exception:
    traceback.print_exc()
    print("CELL 6 FAILED — not skipping silently.")
''')

md("""## Cell 7 — Tier 1 Surya

Tries (a) `surya-ocr==0.14.6` then (b) explicit GQA reshape. **Never** `ignore_mismatched_sizes`. If both fail, writes a not-viable report and stops this cell.""")

code(r'''
print("=" * 72)
print("CELL 7 — TIER 1 SURYA")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
try:
    rc = subprocess.call(
        [sys.executable, "src/probes/run_surya_positive_control.py"],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    print(f"[surya] helper exit {rc}")
    if rc != 0:
        print(
            "Surya positive control is NOT VIABLE on this runtime. "
            "See docs/tier1_surya_control.md. Not forcing a broken load. "
            "Continuing to Cell 8 (demo VLM) which does not need Surya."
        )
    else:
        print("Surya helper reported a load path. Full 60+60 diagnostic is only run if the helper returned 0 and a torch model exists.")
        print("If the written doc still says not viable, treat Cell 7 as stopped.")
except Exception:
    traceback.print_exc()
    print("CELL 7 FAILED — not skipping silently. Cell 8 is independent.")
''')

md("""## Cell 8 — Tier 2 Step 1: inspect + LoRA VRAM (closes Decision #3)

`--inspect` first (real module names). Then dummy LoRA train steps. Writes `docs/demo_lora_inspect.json` and `docs/demo_lora_vram.json`.""")

code(r'''
print("=" * 72)
print("CELL 8 — DEMO BACKBONE T4 VRAM")
print("=" * 72)
from pathlib import Path
import os, json, subprocess, sys, traceback, gc
import torch

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from colab.notebook_support import append_decision_3_close, choose_demo_backbone

CANDIDATES = [
    "ds4sd/SmolDocling-256M-preview",
    "lightonai/LightOnOCR-1B-1025",
    "ibm-granite/granite-docling-258M",
    "lightonai/LightOnOCR-2-1B-base",
]
inspect_ok = {}
CELL8_OK = False
SELECTED = None
try:
    for mid in CANDIDATES:
        print(f"\n----- INSPECT {mid} -----")
        try:
            subprocess.check_call(
                [sys.executable, "src/models/demo/benchmark_base_models.py",
                 "--model-id", mid, "--inspect"],
                env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            )
            inspect_ok[mid] = True
        except subprocess.CalledProcessError as e:
            print(f"INSPECT FAILED for {mid}: {e}")
            inspect_ok[mid] = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    inspect_path = REPO / "docs" / "demo_lora_inspect.json"
    payload = json.loads(inspect_path.read_text()) if inspect_path.exists() else {}
    vram_path = REPO / "docs" / "demo_lora_vram.json"
    for mid, ok in inspect_ok.items():
        if not ok:
            continue
        rec = payload.get(mid) or {}
        targets = rec.get("target_modules") or []
        if not targets:
            print(f"skip LoRA dummy for {mid}: no target_modules from inspect")
            continue
        print(f"\n----- LORA DUMMY {mid} targets={targets} -----")
        try:
            subprocess.check_call(
                [sys.executable, "src/models/demo/benchmark_base_models.py",
                 "--model-id", mid, "--target-modules", *targets,
                 "--batch-size", "1", "--steps", "3",
                 "--json-out", str(vram_path)],
                env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            )
        except subprocess.CalledProcessError as e:
            print(f"VRAM DUMMY FAILED for {mid}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vram = json.loads(vram_path.read_text()) if vram_path.exists() else {}
    choice = choose_demo_backbone(vram)
    print("DECISION #3 CALL:", json.dumps(choice, indent=2, default=str))
    (REPO / "docs" / "demo_base_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    if choice.get("selected"):
        append_decision_3_close(REPO, choice)
        SELECTED = choice["selected"]
        CELL8_OK = True
        summary = [
            "# Tier 2 / Stage 2b — T4 LoRA benchmark (Colab)",
            "",
            f"**Selected:** `{SELECTED}`",
            "",
            f"**Why:** {choice.get('reason')}",
            "",
            "Peaks:",
            "",
        ]
        for mid, rec in vram.items():
            summary.append(
                f"- `{mid}`: peak {rec.get('peak_gb')} GB; "
                f"fits_14GB_headroom={rec.get('fits_t4_14gb_headroom')}; "
                f"headroom {rec.get('headroom_gb')} GB"
            )
        (REPO / "docs" / "tier2_stage2b_demo.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("Wrote docs/tier2_stage2b_demo.md")
    else:
        print("NO MODEL SELECTED. Cell 9 will skip SFT.")
        (REPO / "docs" / "tier2_stage2b_demo.md").write_text(
            "# Tier 2 / Stage 2b — T4 LoRA benchmark\n\n"
            "No candidate stayed under 14 GB. Decision #3 still open.\n",
            encoding="utf-8",
        )
    print("Cell 8 finished. CELL8_OK =", CELL8_OK, "SELECTED =", SELECTED)
except Exception:
    traceback.print_exc()
    print("CELL 8 FAILED — Cell 9 must not pretend SFT is allowed.")
    CELL8_OK = False
    SELECTED = None
''')

md("""## Cell 9 — Tier 2 Step 2: LoRA SFT (only if Cell 8 selected a model)

Checkpoints under `checkpoints/demo/`. Partial runs are reported as partial.""")

code(r'''
print("=" * 72)
print("CELL 9 — DEMO LORA SFT")
print("=" * 72)
from pathlib import Path
import os, json, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
choice_path = REPO / "docs" / "demo_base_choice.json"
CELL9_OK = False
if not choice_path.exists():
    print("SKIP Cell 9: Cell 8 did not write demo_base_choice.json")
else:
    choice = json.loads(choice_path.read_text())
    selected = choice.get("selected")
    if not selected:
        print("SKIP Cell 9: no backbone selected. Reason:", choice.get("reason"))
    else:
        try:
            cmd = [
                sys.executable, "src/models/demo/sft.py",
                "--run", "--model-id", selected, "--device", "cuda",
                "--manifest", "data/manifests/hindi_natural.jsonl",
                "--data-root", "data",
                "--output-root", "checkpoints/demo",
                "--max-steps", "100", "--ckpt-every", "20",
            ]
            print("[run]", " ".join(cmd))
            subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
            adapter = REPO / "checkpoints" / "demo" / "adapter_config.json"
            fail = REPO / "checkpoints" / "demo" / "sft_not_trained.json"
            if adapter.exists():
                CELL9_OK = True
                print("SFT adapter present at", adapter)
            elif fail.exists():
                print("SFT STOPPED (collate/error). Marker:")
                print(fail.read_text())
            else:
                print("SFT process returned but no adapter_config.json — treat as incomplete.")
            extra = REPO / "docs" / "tier2_stage2b_demo.md"
            extra.write_text(
                extra.read_text(encoding="utf-8")
                + f"\n## SFT\n\nselected={selected}\nadapter={adapter.exists()}\n"
                + (fail.read_text() if fail.exists() else "no sft_not_trained.json\n"),
                encoding="utf-8",
            )
        except Exception:
            traceback.print_exc()
            print("CELL 9 FAILED. Cell 10 will skip RLVR.")
print("CELL9_OK =", CELL9_OK)
''')

md("""## Cell 10 — RLVR coverage-term ablation (only if Cell 9 produced an adapter)

One run, λ_coverage=0. Not a reward sweep.""")

code(r'''
print("=" * 72)
print("CELL 10 — RLVR λ=0")
print("=" * 72)
from pathlib import Path
import os, subprocess, sys, traceback

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
adapter = REPO / "checkpoints" / "demo" / "adapter_config.json"
if not adapter.exists():
    print("SKIP RLVR: no SFT adapter. Not attempting a truncated train.")
    (REPO / "docs" / "tier2_rlvr_ablation.md").write_text(
        "# Tier 2 — RLVR coverage ablation\n\n"
        "**Not attempted.** Cell 9 did not produce `checkpoints/demo/adapter_config.json`.\n",
        encoding="utf-8",
    )
else:
    try:
        cmd = [
            sys.executable, "src/models/demo/rlvr_train.py",
            "--sft-root", "checkpoints/demo",
            "--lambda-coverage", "0.0",
            "--max-steps", "30",
            "--device", "cuda",
            "--output-root", "checkpoints/demo_rlvr_nocov",
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        summ = REPO / "checkpoints" / "demo_rlvr_nocov" / "rlvr_summary.json"
        body = "# Tier 2 — RLVR coverage ablation (Colab)\n\n"
        if summ.exists():
            body += summ.read_text()
        (REPO / "docs" / "tier2_rlvr_ablation.md").write_text(body + "\n", encoding="utf-8")
        print("Wrote docs/tier2_rlvr_ablation.md")
    except Exception:
        traceback.print_exc()
        print("CELL 10 FAILED.")
''')

md("""## Cell 11 — PAUSE. Read the diff. Do **not** commit in this cell.

The next cell is the only one that `git commit` / `git push`.""")

code(r'''
print("=" * 72)
print("CELL 11 — DIFF PREVIEW (no commit)")
print("=" * 72)
from pathlib import Path
import subprocess, os

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
subprocess.call(["git", "status"])
print("\n----- git diff --stat -----\n")
subprocess.call(["git", "diff", "--stat"])
print("\n----- git diff (docs/ and DECISIONS.md) -----\n")
subprocess.call(["git", "diff", "--", "docs/", "DECISIONS.md"])
print("=" * 72)
print("If this looks right, run Cell 12 yourself. It will add, commit, and push.")
print("paper/main.tex is not included.")
print("=" * 72)
''')

md("""## Cell 12 — commit + push (run only after reading Cell 11)

Needs a GitHub token in Colab. Will not run unless you execute this cell.""")

code(r'''
print("=" * 72)
print("CELL 12 — COMMIT + PUSH")
print("=" * 72)
from pathlib import Path
import os, subprocess

REPO = Path("/content/vlm-ocr-eval")
os.chdir(REPO)
subprocess.check_call(["git", "config", "user.email", "colab@local"])
subprocess.check_call(["git", "config", "user.name", "colab-t4-notebook"])
subprocess.check_call([
    "git", "add",
    "docs/",
    "data/probe_results/",
    "DECISIONS.md",
])
print("staged:")
subprocess.check_call(["git", "status"])
msg = "Colab T4: Tier 0 probes, Surya control attempt, demo LoRA VRAM / SFT / RLVR"
subprocess.check_call(["git", "commit", "-m", msg])
print("Commit created. Push needs a token.")
print("Set env GH_TOKEN to a PAT with repo scope, then this cell will push.")
token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
if not token:
    print("NO TOKEN — commit is local to this VM only. Download docs/ if the session will die.")
else:
    subprocess.check_call([
        "git", "push",
        f"https://x-access-token:{token}@github.com/Adya6714/vlm-ocr-eval.git",
        "HEAD:main",
    ])
    print("Pushed to origin/main")
''')

out = Path("notebooks/colab_run.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
# fix source arrays: last cell writer used split which is fine
out.write_text(json.dumps(NB, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(NB["cells"]))

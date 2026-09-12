"""
LoRA adapter config for the demo VLM.

Why a separate file: target_modules are model-specific. Decision #3's
benchmark script exists so we copy names from --inspect instead of
guessing q_proj on an architecture that uses q_lin.

Until inspect json exists, TARGET_MODULES_BY_MODEL is empty on purpose
for uninspected ids — sft.py must refuse to train rather than attach
LoRA to the wrong leaves.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
INSPECT_JSON = _REPO / "docs" / "demo_lora_inspect.json"

# Filled only after a real --inspect run writes demo_lora_inspect.json.
# Do not invent q_proj lists here.
TARGET_MODULES_BY_MODEL: dict[str, list[str]] = {}


def load_inspected_targets(model_id: str, path: Path | None = None) -> list[str]:
    path = path or INSPECT_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"no inspect dump at {path}; run benchmark_base_models.py "
            "--inspect and write leaf names before SFT"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rec = payload.get(model_id)
    if not rec or not rec.get("target_modules"):
        raise KeyError(
            f"{model_id} missing target_modules in {path}; inspect that id first"
        )
    return list(rec["target_modules"])


def build_lora_config(
    model_id: str,
    *,
    r: int = 16,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> "LoraConfig":
    """
    PEFT LoRA on inspected attention projections only.

    r=16 is a T4-friendly starting rank, not a tuned hyperparameter.
    """
    modules = target_modules or TARGET_MODULES_BY_MODEL.get(model_id)
    if not modules:
        modules = load_inspected_targets(model_id)
    if lora_alpha is None:
        lora_alpha = 2 * r
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

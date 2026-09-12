"""PEFT target_modules must be Linear/Embedding/Conv leaves, not containers."""

import sys
import unittest
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models" / "demo"))

from benchmark_base_models import collect_peft_target_leaves  # noqa: E402


class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttn()
        self.modality_projection = nn.Sequential(nn.Linear(8, 8))
        self.multi_modal_projector = nn.Module()  # empty container


class PeftLeafTests(unittest.TestCase):
    def test_drops_attn_and_projector_containers(self):
        leaves, dotted = collect_peft_target_leaves(FakeVLM())
        self.assertNotIn("self_attn", leaves)
        self.assertNotIn("modality_projection", leaves)
        self.assertNotIn("multi_modal_projector", leaves)
        self.assertIn("q_proj", leaves)
        self.assertIn("k_proj", leaves)
        self.assertIn("v_proj", leaves)
        self.assertIn("self_attn.q_proj", dotted)
        self.assertFalse(any(p.rsplit(".", 1)[-1] in {
            "self_attn", "modality_projection", "multi_modal_projector",
        } for p in dotted))


if __name__ == "__main__":
    unittest.main()

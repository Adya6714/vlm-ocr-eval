"""PEFT target_modules must be Linear/Embedding/Conv leaves, not containers."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models" / "demo"))

from benchmark_base_models import (  # noqa: E402
    collect_peft_target_leaves,
    dummy_mistral3_input_ids,
    dummy_vision_batch,
)


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


class DummyVisionBatchTests(unittest.TestCase):
    def test_idefics3_is_five_dim(self):
        pix, kw = dummy_vision_batch(
            "idefics3", batch_size=2, image_size=32, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(tuple(pix.shape), (2, 1, 3, 32, 32))
        self.assertEqual(kw, {})

    def test_mistral3_passes_image_sizes_hw(self):
        pix, kw = dummy_vision_batch(
            "mistral3", batch_size=2, image_size=28, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(tuple(pix.shape), (2, 3, 28, 28))
        sizes = kw["image_sizes"]
        self.assertEqual(tuple(sizes.shape), (2, 2))
        self.assertEqual(sizes.dtype, torch.long)
        self.assertEqual(sizes.tolist(), [[28, 28], [28, 28]])

    def test_unknown_family_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            dummy_vision_batch(
                "not_a_family", batch_size=1, image_size=16, device=torch.device("cpu"), dtype=torch.float32
            )
        self.assertIn("unrecognized model_type", str(ctx.exception))


class Mistral3PlaceholderIdsTests(unittest.TestCase):
    def test_count_matches_and_grows_seq(self):
        ids = dummy_mistral3_input_ids(
            batch_size=2,
            seq_len=8,
            n_image_tokens_per_example=20,
            image_token_id=151655,
            device=torch.device("cpu"),
        )
        self.assertEqual(ids.shape[1], 21)
        self.assertEqual(int((ids == 151655).sum()), 40)
        self.assertTrue(torch.all(ids[:, 20] != 151655))


if __name__ == "__main__":
    unittest.main()

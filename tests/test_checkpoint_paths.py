"""Checkpoint/tokenizer path isolation and resume safety guards."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_DIR = ROOT / "src" / "models" / "instrument"
sys.path.insert(0, str(INSTRUMENT_DIR))

from train import (  # noqa: E402
    checkpoint_path,
    tokenizer_path,
    verify_checkpoint_matches_run,
    verify_tokenizer_matches_checkpoint,
)
from tokenizer import GraphemeTokenizer  # noqa: E402


class TestCheckpointPaths(unittest.TestCase):
    def test_hindi_bengali_checkpoint_paths_differ(self):
        root = "/tmp/out"
        hindi = checkpoint_path(root, "hindi", "natural", 0)
        bengali = checkpoint_path(root, "bengali", "natural", 0)
        self.assertNotEqual(hindi, bengali)
        self.assertEqual(hindi, os.path.join(root, "checkpoint_hindi_natural_seed0.pt"))
        self.assertEqual(bengali, os.path.join(root, "checkpoint_bengali_natural_seed0.pt"))

    def test_hindi_bengali_tokenizer_paths_differ(self):
        root = "/tmp/out"
        hindi = tokenizer_path(root, "hindi", "natural")
        bengali = tokenizer_path(root, "bengali", "natural")
        self.assertNotEqual(hindi, bengali)
        self.assertEqual(hindi, os.path.join(root, "tokenizer_hindi_natural.json"))
        self.assertEqual(bengali, os.path.join(root, "tokenizer_bengali_natural.json"))

    def test_resume_rejects_mismatched_script(self):
        ckpt = {
            "model_state": {"decoder.token_embed.weight": torch.zeros(10, 384)},
            "step": 100,
            "script": "bengali",
            "condition": "natural",
            "seed": 0,
        }
        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_matches_run(ckpt, "hindi", "natural")
        self.assertIn("script", str(ctx.exception).lower())
        self.assertIn("refusing to resume", str(ctx.exception).lower())

    def test_resume_rejects_mismatched_condition(self):
        ckpt = {
            "model_state": {"decoder.token_embed.weight": torch.zeros(10, 384)},
            "step": 100,
            "script": "hindi",
            "condition": "flattened",
            "seed": 0,
        }
        with self.assertRaises(ValueError):
            verify_checkpoint_matches_run(ckpt, "hindi", "natural")

    def test_tokenizer_vocab_mismatch_raises(self):
        tok = GraphemeTokenizer()
        tok.cluster_to_id = {"<PAD>": 0, "अ": 1}
        tok.id_to_cluster = {0: "<PAD>", 1: "अ"}
        ckpt = {
            "model_state": {"decoder.token_embed.weight": torch.zeros(50, 384)},
        }
        with self.assertRaises(ValueError) as ctx:
            verify_tokenizer_matches_checkpoint(tok, ckpt)
        self.assertIn("vocabulary size mismatch", str(ctx.exception).lower())

    def test_mismatched_script_checkpoint_on_disk_raises_on_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = checkpoint_path(tmp, "hindi", "natural", 0)
            torch.save(
                {
                    "model_state": {"decoder.token_embed.weight": torch.zeros(10, 384)},
                    "optimizer_state": {},
                    "step": 200,
                    "script": "bengali",
                    "condition": "natural",
                    "seed": 0,
                },
                path,
            )
            loaded = torch.load(path, map_location="cpu")
            with self.assertRaises(ValueError):
                verify_checkpoint_matches_run(loaded, "hindi", "natural")


if __name__ == "__main__":
    unittest.main()

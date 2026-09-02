"""
Training loop for the instrument model (Stage 2a).

INTERFACE DEPENDENCY, read before using: this expects a "manifest"
JSONL file with one line-level training example per row:
    {"image_path": "path/to/line_crop.png", "text": "ground truth string"}

Stage 1's renderer (src/renderer/) needs to produce this, or a small
adapter script needs to convert its page-level output + bounding boxes
into line crops in this shape. This was flagged as an open question in
docs/stage2_design_notes.md rather than assumed silently -- confirm
with whoever's building Stage 1 before pointing this at real data.

Wires encoder.py + decoder.py together (they have different hidden
dims by design, see decoder.py's docstring -- a projection layer
bridges them here). Implements AGENTS.md's resumability standard:
checkpoints every N steps, resumes automatically on restart, prints
progress per N steps rather than only at start/end.

fp16 (not bf16) throughout -- Colab T4 is Turing architecture, no
bf16 support (IMPLEMENTATION.md, DECISIONS.md #2).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from tokenizer import GraphemeTokenizer
from encoder import InstrumentEncoder
from decoder import InstrumentDecoder

PATCH_SIZE = 14


class InstrumentModel(nn.Module):
    """
    Combines encoder + decoder into one trainable model. Owns the
    dimension bridge between them (encoder d_model=320, decoder
    d_model=384 per docs/stage2_design_notes.md's sizing -- they're
    deliberately different, see decoder.py) via a single linear
    projection.
    """

    def __init__(self, vocab_size: int, encoder_dim: int = 320, decoder_dim: int = 384):
        super().__init__()
        self.encoder = InstrumentEncoder(d_model=encoder_dim)
        self.memory_projection = nn.Linear(encoder_dim, decoder_dim)
        self.decoder = InstrumentDecoder(vocab_size=vocab_size, d_model=decoder_dim)

    def forward(self, images, target_ids, image_padding_mask=None, target_padding_mask=None):
        memory = self.encoder(images, padding_mask=image_padding_mask)
        memory = self.memory_projection(memory)
        logits = self.decoder(
            target_ids, memory,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=image_padding_mask,
        )
        return logits


class LineDataset(Dataset):
    """
    Reads the manifest described in this file's module docstring.
    Images are loaded grayscale and padded to a width divisible by
    PATCH_SIZE at collate time, not here -- padding needs to happen
    per-batch (to the batch's max width), not per-image.
    """

    def __init__(self, manifest_path: str, tokenizer: GraphemeTokenizer):
        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("L")  # grayscale
        token_ids = self.tokenizer.encode(row["text"])
        return image, token_ids


def collate_batch(batch, pad_id: int):
    """
    Pads a batch of (variable-width image, variable-length token list)
    pairs to common sizes, and builds the padding masks the encoder
    and decoder need to ignore that padding in attention.

    Image padding: pad width up to the batch max, THEN round up to the
    next multiple of PATCH_SIZE (patch embedding requires this).
    Height is assumed fixed across a batch (renderer should produce
    fixed-height line crops, e.g. 64px, per docs/stage2_design_notes.md).
    """
    images, token_lists = zip(*batch)

    max_width = max(img.width for img in images)
    max_width = ((max_width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE  # round up
    height = images[0].height

    image_tensors = []
    image_padding_masks = []
    for img in images:
        arr = torch.from_numpy(
            __import__("numpy").array(img, dtype="float32") / 255.0
        ).unsqueeze(0)  # [1, H, W]
        pad_amount = max_width - img.width
        arr = F.pad(arr, (0, pad_amount), value=1.0)  # pad with white (background)
        image_tensors.append(arr)

        num_patches_total = (height // PATCH_SIZE) * (max_width // PATCH_SIZE)
        num_patches_real = (height // PATCH_SIZE) * (img.width // PATCH_SIZE)
        mask = torch.zeros(num_patches_total, dtype=torch.bool)
        mask[num_patches_real:] = True  # True = padding, ignored by attention
        image_padding_masks.append(mask)

    images_batch = torch.stack(image_tensors)             # [B, 1, H, max_width]
    image_padding_batch = torch.stack(image_padding_masks)  # [B, num_patches]

    max_len = max(len(t) for t in token_lists)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_padding = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, tokens in enumerate(token_lists):
        target_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        target_padding[i, :len(tokens)] = False

    return images_batch, image_padding_batch, target_ids, target_padding


SCRIPT_CHOICES = ("hindi", "bengali")


def checkpoint_path(output_root: str, script: str, condition: str, seed: int) -> str:
    """
    One checkpoint path per (script, condition, seed) triple -- Probe 1
    runs 9 independent training runs per script (3 conditions x 3
    seeds, DECISIONS.md #14), and Hindi/Bengali runs must never share
    a path even when condition/seed match (DECISIONS.md #47). This
    function is the single source of truth for where a given run's
    checkpoint lives, used both when saving and when checking for a
    resume on startup.

    Overwritten every args.checkpoint_every steps during training;
    intermediate weights are lost unless --keep-snapshots is set
    (DECISIONS.md #48).
    """
    return os.path.join(output_root, f"checkpoint_{script}_{condition}_seed{seed}.pt")


def snapshot_checkpoint_path(
    output_root: str, script: str, condition: str, seed: int, step: int,
) -> str:
    """
    Immutable per-step weight snapshot for Probe 3's training-curve
    analysis. Only written when train() is called with
    --keep-snapshots; the main checkpoint_path() file is still
    overwritten for resume.
    """
    return os.path.join(
        output_root, f"checkpoint_{script}_{condition}_seed{seed}_step{step}.pt",
    )


def tokenizer_path(output_root: str, script: str, condition: str) -> str:
    """
    One tokenizer file per (script, condition) pair. Vocabulary is built
    from that script's manifest, so Hindi and Bengali must not share a
    tokenizer path (they have different grapheme inventories).
    """
    return os.path.join(output_root, f"tokenizer_{script}_{condition}.json")


def checkpoint_vocab_size(ckpt: dict) -> int:
    """Embedding rows in the saved decoder -- must match len(tokenizer)."""
    return ckpt["model_state"]["decoder.token_embed.weight"].shape[0]


def verify_checkpoint_matches_run(ckpt: dict, script: str, condition: str) -> None:
    """
    Refuse to resume when checkpoint metadata disagrees with the current
    run. Silent cross-script resume caused real data loss (Bengali
    tokenizer overwrote Hindi at the old shared path).
    """
    ckpt_script = ckpt.get("script")
    ckpt_condition = ckpt.get("condition")
    if ckpt_script != script or ckpt_condition != condition:
        raise ValueError(
            f"checkpoint/run mismatch: checkpoint has script={ckpt_script!r} "
            f"condition={ckpt_condition!r}, but this run requested "
            f"script={script!r} condition={condition!r}. "
            f"Refusing to resume — loading a checkpoint from a different "
            f"script or condition would silently corrupt training."
        )


def verify_tokenizer_matches_checkpoint(tokenizer: GraphemeTokenizer, ckpt: dict) -> None:
    """
    Catch tokenizer/checkpoint pairs from different runs before
    load_state_dict raises an opaque embedding shape error.
    """
    ckpt_vocab = checkpoint_vocab_size(ckpt)
    tok_vocab = len(tokenizer)
    if tok_vocab != ckpt_vocab:
        raise ValueError(
            f"tokenizer/checkpoint vocabulary size mismatch: tokenizer has "
            f"{tok_vocab} entries but checkpoint embeddings expect "
            f"{ckpt_vocab}. These are from different runs — the tokenizer "
            f"and checkpoint must come from the same training run."
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = checkpoint_path(args.output_root, args.script, args.condition, args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    # --- tokenizer: build fresh, or load if resuming (AGENTS.md
    # resumability -- a resumed run must use the SAME vocabulary,
    # not a re-built one that could assign different ids) ---
    tok_path = tokenizer_path(args.output_root, args.script, args.condition)
    if os.path.exists(tok_path):
        print(f"loading existing tokenizer -> {tok_path}")
        tokenizer = GraphemeTokenizer.load(tok_path)
    else:
        print("building tokenizer from manifest...")
        texts = [json.loads(line)["text"] for line in open(args.manifest, encoding="utf-8")]
        tokenizer = GraphemeTokenizer()
        tokenizer.build_vocab(texts, min_freq=5)
        tokenizer.save(tok_path)
    print(f"vocab size: {len(tokenizer)}")

    dataset = LineDataset(args.manifest, tokenizer)
    pad_id = tokenizer.cluster_to_id["<PAD>"]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
    )

    torch.manual_seed(args.seed)
    model = InstrumentModel(vocab_size=len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))  # fp16, not bf16 -- T4 has no bf16

    start_step = 0
    if os.path.exists(ckpt_path):
        print(f"resuming from checkpoint -> {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        verify_checkpoint_matches_run(ckpt, args.script, args.condition)
        verify_tokenizer_matches_checkpoint(tokenizer, ckpt)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        print(f"resumed at step {start_step}")
    else:
        print("no checkpoint found, starting fresh")

    model.train()
    step = start_step
    t0 = time.time()

    while step < args.total_steps:
        for images, image_padding, target_ids, target_padding in loader:
            if step >= args.total_steps:
                break

            images = images.to(device)
            image_padding = image_padding.to(device)
            target_ids = target_ids.to(device)
            target_padding = target_padding.to(device)

            # teacher forcing: input is target shifted right by one,
            # loss is computed against the actual next token
            decoder_input = target_ids[:, :-1]
            decoder_input_padding = target_padding[:, :-1]
            labels = target_ids[:, 1:]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(images, decoder_input, image_padding, decoder_input_padding)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=pad_id,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1

            # progress printing per AGENTS.md's new standard -- not
            # just at start/end, so a slow-but-working run is
            # distinguishable from a hung one
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = args.log_every / elapsed if elapsed > 0 else 0
                remaining = (args.total_steps - step) / steps_per_sec if steps_per_sec > 0 else float("inf")
                print(f"[{args.script} {args.condition} seed={args.seed}] step {step}/{args.total_steps}  "
                      f"loss={loss.item():.4f}  {steps_per_sec:.2f} steps/s  "
                      f"ETA {remaining/60:.1f} min")
                t0 = time.time()

            # checkpointing per AGENTS.md's resumability standard --
            # save regularly, not just at the end, so a killed Colab
            # session loses at most args.checkpoint_every steps of work
            if step % args.checkpoint_every == 0:
                ckpt_payload = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "loss": float(loss.item()),
                    "script": args.script,
                    "condition": args.condition,
                    "seed": args.seed,
                }
                torch.save(ckpt_payload, ckpt_path)
                print(f"  checkpoint saved -> {ckpt_path} (step {step})")
                if getattr(args, "keep_snapshots", False):
                    snap_path = snapshot_checkpoint_path(
                        args.output_root, args.script, args.condition, args.seed, step,
                    )
                    torch.save(ckpt_payload, snap_path)
                    print(f"  snapshot saved -> {snap_path}")

    print(f"training complete: {step} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="JSONL manifest: one {'image_path', 'text'} per line")
    parser.add_argument("--script", required=True, choices=list(SCRIPT_CHOICES),
                         help="Writing system this run trains on (hindi or bengali)")
    parser.add_argument("--condition", required=True,
                         choices=["natural", "flattened", "inverted"],
                         help="Probe 1 glyph-frequency condition this run corresponds to")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="checkpoints",
                         help="Colab-friendly single root for checkpoints + tokenizer -- "
                              "point this at a Drive-mounted path when running on Colab")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--keep-snapshots", action="store_true",
                         help="Also write immutable per-step snapshot files "
                              "(checkpoint_{script}_{condition}_seed{seed}_step{N}.pt) "
                              "for probe3_training_curve.py. Default off to save Drive space.")
    args = parser.parse_args()
    train(args)

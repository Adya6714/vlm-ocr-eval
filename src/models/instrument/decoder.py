"""
Small autoregressive decoder for the instrument model (Stage 2a).

Generates grapheme-cluster tokens one at a time, cross-attending to
the encoder's image features (encoder.py) at every step -- this is the
"reading" half described in BOOK.md Chapter 0: given the image and
everything generated so far, predict what comes next.

Sizing: docs/stage2_design_notes.md -- 5 layers, hidden dim 384,
larger than the encoder's on purpose (the decoder does more of the
actual work at this model size). ~13-14M params including the tied
embedding/output head.
"""

import torch
import torch.nn as nn

from tokenizer import GraphemeTokenizer  # local import; both files live in src/models/instrument/


class InstrumentDecoder(nn.Module):
    """
    Token embedding + learned positional embedding + a stack of
    transformer decoder layers (self-attention with a causal mask,
    cross-attention to encoder memory, feed-forward) + an output head
    tied to the input embedding.

    Positional embedding is LEARNED here, unlike the encoder's
    sinusoidal choice -- decoder sequences (grapheme clusters in one
    line) have a bounded, predictable max length, unlike variable-width
    images, so a learned table is affordable and slightly more
    expressive.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 5,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,  # generous for one rendered line of grapheme clusters
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(layer, num_layers=num_layers)

        # output head tied to token_embed's weights -- halves the
        # parameter cost of the vocabulary (see tokenizer.py's design
        # notes reference); the same matrix is used to look up an
        # embedding AND to score every vocabulary entry at output time.
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

    def forward(
        self,
        target_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        target_padding_mask: torch.Tensor = None,
        memory_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        target_ids: [B, T] token ids generated so far (during training,
                    this is the ground-truth sequence shifted right --
                    "teacher forcing", see BOOK.md Chapter 3 once written)
        encoder_memory: [B, num_patches, d_model_encoder] -- output of
                        InstrumentEncoder. NOTE: encoder d_model (320)
                        and decoder d_model (384) differ per the design
                        doc's sizing; a projection is needed at the
                        point these are wired together in train.py.
        target_padding_mask: [B, T] bool, True where target_ids is padding
        memory_padding_mask: [B, num_patches] bool, True where encoder
                              output is padding (same mask the encoder used)
        returns: [B, T, vocab_size] logits over the vocabulary at each
                 position
        """
        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(target_ids) + self.pos_embed(positions)

        # causal mask: position i can attend to positions <= i only.
        # This is what makes generation autoregressive rather than
        # "cheating" by seeing the answer during training.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=target_ids.device)

        x = self.layers(
            tgt=x,
            memory=encoder_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.output_head(x)
        return logits


if __name__ == "__main__":
    # Smoke test with a tiny real tokenizer + fake encoder memory --
    # confirms the causal mask actually blocks future information
    # before this touches real training data.
    torch.manual_seed(0)

    tok = GraphemeTokenizer()
    tok.build_vocab(["हिन्दी एक भाषा है।"] * 5, min_freq=1)
    vocab_size = len(tok)
    print(f"tiny vocab size: {vocab_size}")

    decoder = InstrumentDecoder(vocab_size=vocab_size, d_model=384, num_layers=5)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"decoder parameter count: {n_params:,}")

    batch_size, seq_len, num_patches, encoder_dim = 2, 10, 50, 384
    fake_target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    fake_memory = torch.randn(batch_size, num_patches, encoder_dim)

    logits = decoder(fake_target_ids, fake_memory)
    print(f"output shape: {logits.shape}  (expect [2, 10, {vocab_size}])")
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # causal-mask check: changing a LATER token must not change an
    # EARLIER position's logits, since that would mean the model is
    # illegally attending to the future. Must run in eval() mode --
    # dropout is stochastic per forward call and would fail this check
    # for reasons that have nothing to do with the causal mask.
    decoder.eval()
    with torch.no_grad():
        modified_ids = fake_target_ids.clone()
        modified_ids[:, -1] = (modified_ids[:, -1] + 1) % vocab_size
        logits_modified = decoder(modified_ids, fake_memory)
        logits_recheck = decoder(fake_target_ids, fake_memory)  # re-run original too, for a fair comparison under eval()
    early_positions_unchanged = torch.allclose(
        logits_recheck[:, :-1, :], logits_modified[:, :-1, :], atol=1e-5
    )
    print(f"causal mask respected (early positions unaffected by later token change): {early_positions_unchanged}")
    assert early_positions_unchanged, "causal mask is leaking future information -- do not proceed"
    print("all decoder checks OK")

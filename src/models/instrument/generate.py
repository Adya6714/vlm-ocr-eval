"""
Autoregressive generation for the instrument model.

Why this exists: train.py only computes loss via teacher forcing
(scoring the model against a known-correct sequence) -- it never
actually generates text from a new image. Every probe that reads a
fresh image needs this:
    Probe 2 (confusion structure) needs the full output distribution
             at each step, not just the argmax prediction.
    Probe 3 (blank/noise control) needs predicted text on images with
             no real content.
    Probe 5 (calibration) needs per-step confidence alongside each
             prediction.
    Attention ablation (Claim B mechanism) needs a zero-encoder-memory
             path and optional teacher-forced prefixes so per-step KL
             is not confounded by diverging greedy sequences
             (DECISIONS.md #56).

Greedy decoding only (no beam search, no sampling) -- this is a
diagnostic instrument, not a production reading system; greedy is
simpler, deterministic (same input always gives the same output, which
matters for reproducible probe results), and sufficient for measuring
what the model learned rather than optimizing output quality.

Not KV-cached (recomputes the full decoder forward pass at every step
rather than caching past attention keys/values). This is a real
inefficiency, acceptable at this model's small scale (max_len=128,
~12M decoder params) -- flagged here rather than silently accepted, in
case a later stage needs more speed than this provides.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tokenizer import GraphemeTokenizer


@torch.no_grad()
def generate(
    model,
    image: torch.Tensor,
    tokenizer: GraphemeTokenizer,
    max_len: int = 128,
    device: torch.device = None,
    zero_encoder_memory: bool = False,
    return_full_probs: bool = False,
    force_next_ids: list | None = None,
) -> dict:
    """
    Generates text for ONE image (batch size 1 -- probes process
    images individually since they need per-image distributions, not
    batch-averaged ones).

    image: [1, C, H, W], already preprocessed (grayscale, normalized)
           the same way train.py's collate_batch prepares training
           images -- callers are responsible for matching that
           preprocessing exactly, or results won't be comparable to
           training-time behavior.

    Ablation knobs (attention_ablation / Claim B mechanism probe):
        zero_encoder_memory: replace the encoder's output with an
            all-zeros tensor of identical shape *before*
            memory_projection. The decoder still runs its own
            autoregressive prior and still cross-attends, but the
            keys/values carry no image information — only the
            projected zero vector. That is the "prior-only" condition.
        force_next_ids: if provided, append these token ids instead of
            the greedy argmax (teacher forcing). Used so the zero-memory
            pass can be scored under the *same prefixes* as the
            full-memory generation, making per-step KL / top-1
            agreement a clean memory-vs-prior comparison rather than a
            confound of diverging sequences.
        return_full_probs: include the full vocab softmax per step.
            Expensive to persist; probes that need it usually compute
            KL online and drop the vectors (DECISIONS.md #56).

    Returns a dict, not just a string, because every probe needs
    different pieces of this:
        "text": the decoded string (strip special tokens)
        "token_ids": the raw generated id sequence
        "step_confidences": list of float, max softmax probability at
            each generated step -- this IS the calibration signal
            Probe 5 measures.
        "step_top_k": list of [(cluster, probability), ...] per step,
            top 5 candidates -- this is what Probe 2's confusion graph
            is built from: not just what the model said, but what it
            almost said instead.
        "step_probs": (optional) list of 1-d probability tensors / lists
            when return_full_probs is True.
        "zero_encoder_memory": bool echo of the ablation flag.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    image = image.to(device)
    bos_id = tokenizer.cluster_to_id["<BOS>"]
    eos_id = tokenizer.cluster_to_id["<EOS>"]

    memory = model.encoder(image, padding_mask=None)
    # Zero BEFORE projection so the ablation removes image features,
    # not just the projected residual — projection of zeros is still
    # a learned bias path the decoder can use as a constant.
    if zero_encoder_memory:
        memory = torch.zeros_like(memory)
    memory = model.memory_projection(memory)

    generated_ids = [bos_id]
    step_confidences = []
    step_top_k = []
    step_probs = []

    # force_next_ids is the sequence of tokens AFTER <BOS>, one per step.
    n_forced = len(force_next_ids) if force_next_ids is not None else None

    for step_i in range(max_len):
        if n_forced is not None and step_i >= n_forced:
            break
        current_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        logits = model.decoder(current_ids, memory)  # [1, len_so_far, vocab_size]
        next_token_logits = logits[0, -1, :]  # only the newest position matters
        probs = F.softmax(next_token_logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, k=min(5, probs.size(0)))
        step_top_k.append([
            (tokenizer.id_to_cluster.get(int(i), "<RARE>"), float(p))
            for p, i in zip(top_probs.tolist(), top_ids.tolist())
        ])

        if force_next_ids is not None:
            next_id = int(force_next_ids[step_i])
            next_confidence = float(probs[next_id].item())
        else:
            next_id = int(top_ids[0].item())
            next_confidence = float(top_probs[0].item())

        generated_ids.append(next_id)
        step_confidences.append(next_confidence)
        if return_full_probs:
            step_probs.append(probs.detach().cpu())

        if next_id == eos_id:
            break

    out = {
        "text": tokenizer.decode(generated_ids, strip_special=True),
        "token_ids": generated_ids,
        "step_confidences": step_confidences,
        "step_top_k": step_top_k,
        "zero_encoder_memory": zero_encoder_memory,
    }
    if return_full_probs:
        out["step_probs"] = step_probs
    return out


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """
    KL(p || q) for two discrete distributions over the same support.

    Why KL(full || zero) specifically: it asks how much extra bits the
    full-memory predictive distribution needs when scored under the
    prior-only distribution — i.e. how much of p_full is *not*
    explained by the decoder prior. The reverse KL(zero || full) is
    available to callers but not the primary ablation metric
    (DECISIONS.md #56).

    Called from: probe_attention_ablation.py, once per generation step
    under a shared prefix. eps floors both sides so empty bins don't
    produce NaN when a token has mass under p but ~0 under q.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    # Renormalize after flooring so they remain proper distributions.
    p = p / p.sum()
    q = q / q.sum()
    return float(torch.sum(p * (torch.log(p) - torch.log(q))).item())


def prior_sufficiency(p_full: torch.Tensor, p_zero: torch.Tensor) -> float:
    """
    Exact probability-mass overlap: sum_i min(p_full[i], p_zero[i]).

    Why this measure: it is the fraction of full-memory probability
    mass that the zero-memory prior *also* assigns to the same tokens
    — one scalar in [0, 1], equal to 1 − TV(p_full, p_zero). A value
    near 1 means the image-conditioned distribution barely moves mass
    relative to the prior; near 0 means almost all mass moved.

    Alternatives considered (and rejected for the primary report):
    top-K restricted overlap (depends on an arbitrary K) and
    Bhattacharyya coefficient (harder to narrate as "shared mass").
    See DECISIONS.md #56.

    Called from: probe_attention_ablation.py per shared-prefix step.
    """
    return float(torch.sum(torch.minimum(p_full, p_zero)).item())


if __name__ == "__main__":
    # Smoke test against a REAL (if undertrained) checkpoint from the
    # Probe 1 orchestration test, not synthetic tensors -- confirms
    # the whole encoder -> memory -> decoder -> generation chain works
    # end to end, including reading a saved checkpoint back correctly.
    import os
    from train import InstrumentModel

    # Check both the interactive-testing path (/tmp/probe1_test, used
    # when running probe1_exposure.py directly) and the Makefile's
    # smoke-test path (/tmp/probe1_smoke, used by `make probe1-smoke`)
    # -- these are two different entry points that both produce a
    # usable checkpoint, and this test should find whichever exists
    # rather than being silently skipped over a path mismatch.
    candidate_roots = ["/tmp/probe1_smoke", "/tmp/probe1_test"]
    ckpt_path = None
    tokenizer_path = None
    for root in candidate_roots:
        candidate_ckpt = os.path.join(root, "checkpoint_hindi_natural_seed0.pt")
        if os.path.exists(candidate_ckpt):
            ckpt_path = candidate_ckpt
            tokenizer_path = os.path.join(root, "tokenizer_hindi_natural.json")
            break

    if ckpt_path is None:
        print(f"SKIPPED: no checkpoint found in {candidate_roots} -- run "
              f"probe1_exposure.py or `make probe1-smoke` first to produce one.")
    else:
        tokenizer = GraphemeTokenizer.load(tokenizer_path)
        model = InstrumentModel(vocab_size=len(tokenizer))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"loaded checkpoint from step {ckpt['step']}, vocab size {len(tokenizer)}")

        # a blank/white image -- also a first look at what Probe 3
        # will eventually do properly: feed the model nothing real
        # and see what it says anyway.
        blank_image = torch.ones(1, 1, 70, 280)  # matches fake training image shape

        result = generate(model, blank_image, tokenizer, max_len=20)
        print(f"generated text: {result['text']!r}")
        print(f"num steps: {len(result['step_confidences'])}")
        print(f"first-step confidence: {result['step_confidences'][0]:.4f}")
        print(f"first-step top candidates: {result['step_top_k'][0]}")

        assert len(result["step_confidences"]) == len(result["step_top_k"])
        assert all(0.0 <= c <= 1.0 for c in result["step_confidences"])
        print("shape and range checks OK")

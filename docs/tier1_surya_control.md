# Tier 1 — Surya positive control

**Status: diagnostic not run.** The requested suite (teacher-forced log
*p*(GT) by Table 6 buckets, position-0 GT rank, self-generated
max-softmax, blank, encoder-zeroing with Table 5 agree/flip KL) was
**not computed**. Nothing below is a substitute number.

## Two different “Surya”s (must not be conflated)

### A. The engine in Table 1 (this repo’s Stage 0)

`src/eval/run_baselines.py` `_get_surya_recognizer` / `run_surya` uses
**surya-ocr 0.22**: `SuryaInferenceManager` + `RecognitionPredictor`,
full-page HTML text. Exact-match **46.8%** (104/222) in
`docs/paper_defensibility_stats.md` Follow-Up 7 is this engine.

That runtime is a **650M Qwen3.5-style VLM** served by **llama.cpp**
(`surya/inference/backends/llamacpp.py`). Generation is an OpenAI-compatible
chat completion. The backend can request **token logprobs of the tokens
it emitted** (`openai_client.py` `logprobs=True`; stored as a mean
probability on the page result). It does **not** expose:

- the full vocabulary softmax at a decoder step (needed for GT **rank**,
  including P(rank>100));
- encoder memory as a tensor (needed to **zero the encoder** and
  re-score under the full-memory greedy prefix, Table 5 / Decision #56);
- a grapheme-cluster vocabulary (Surya’s tokenizer is the VLM’s, not
  this repo’s 367 clusters).

Per the existing plumbing note (`docs/surya_positive_control.md` item 4)
and this inspection: **encoder-zeroing on 0.22 is unrecoverable** without
replacing llama.cpp with a torch decoder we do not have. Self-generated
mean token-prob is **not** max-softmax over grapheme clusters and is
**not** reported here as if it were items 1–5.

### B. The architecture named in the review request

HuggingFace `vikp/surya_rec` exists. `config.json` says
`architectures: ["LangVisionEncoderDecoderModel"]` with a Swin-style
Donut encoder config and an **MBart** decoder (`add_cross_attention:
true`). `preprocessor_config.json` requires **`SuryaImageProcessor` /
`SuryaProcessor`**, which are **not** in stock `transformers` 5.15.1.

This session did **not** finish loading weights
(`VisionEncoderDecoderModel.from_pretrained('vikp/surya_rec')` was
started; the Hub cache stayed at ~44K, i.e. config-only). Even a
successful load would be a **different model** than Table 1’s 46.8%
engine (Surya v1 rec vs Surya 2 VLM). Using it as the positive control
without saying so would mix two systems.

## Attempted torch load of `vikp/surya_rec` (this session)

Weights are on disk: `checkpoints/surya_v1_rec/model.safetensors`
(1,048,833,584 bytes, 2026-09-12). Stock transformers 5.15.1
`VisionEncoderDecoderModel.from_pretrained` **raises**
`RuntimeError` (`ignore_mismatched_sizes=False`, left that way):

- UNEXPECTED: `decoder.model.decoder.layers.3.moe.experts.{65539…65631}.*`
- MISSING: layer-3 `fc1`/`fc2` (MoE vs dense FFN)
- MISMATCH: self-attn / encoder-attn `k_proj`/`v_proj` 256 vs 1024 (GQA)

Tokenizer: needs sentencepiece (not installed) for the slow tokenizer.
Image processor: `SuryaImageProcessor` is not a transformers class.

No logits, ranks, or ablation numbers were produced from this checkpoint.
Do not load it with `ignore_mismatched_sizes=True` — that would be a
different, randomly re-inited model.

## Vocabulary mismatch (would apply to either path)

Instrument GT is scored with `regex.findall(r"\X", ...)` after Tier 1
NFC (`docs/training_config.md`). Surya v1 uses a large sentencepiece /
MBart vocab (language id tokens in the 655xx range in config). Surya 2
uses the VLM tokenizer.

A clean 1–1 map from a Devanagari grapheme cluster to one Surya token
does **not** exist. The planned approximation (plumbing doc item 2):
decode Surya ids → Unicode → `\X` clusters; teacher-force the **shortest
Surya token prefix** that NFC-matches each cluster; **report skip rate**.
That approximation was never applied because the logit loop never ran.

## Encoder-zeroing protocol (if a torch encoder ever exists)

Must match the instrument (`docs/training_config.md` Step 0a;
`probe_attention_ablation.py` lines 10–18, 150–167):

- mean confidence under **independent** zero-memory greedy;
- KL / top-1 / prior-sufficiency by **re-scoring the zero-memory
  decoder on the full-memory greedy token path**, not on GT and not as
  a free-run.

Not implicit. Not implemented for Surya in this session.

## Is the position-0 collapse-while-confident pattern in Surya?

**Unknown.** Not measured. Do not read Table 1 exact-match as an answer
to that question: 46.8% exact match on scored pages is competence on
**text-bearing** images only, with no blank control and no position-0
softmax.

## What would unblock this

1. A torch Surya decoder whose `forward` returns full logits and whose
   encoder output can be zeroed (old `LangVisionEncoderDecoderModel`
   **plus** `SuryaProcessor`), **or**
2. A llama.cpp / vLLM patch that returns the **full** first-token
   distribution and a zero-visual-encoder condition — not available in
   the 0.22 public API.

Until then items (1)–(5) of the task spec stay **not computed**.

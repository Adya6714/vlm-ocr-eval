# PaddleOCR as a positive control — feasibility only

**Status: not viable for the instrument diagnostic suite.** No diagnostic
code was written. Nothing below is a Table 6 / Table 5 number for
PaddleOCR.

This is the same stop rule as Surya (`docs/tier1_surya_control.md`): no
forced loads, no silent approximations, no “close enough” CTC timestep
treated as a grapheme decoder step.

Inspected **installed** packages, not docs-only:

- `paddleocr` **3.7.0** (`paddleocr/__init__.py`)
- `paddlex` text-recognition predictor + processors
- Hugging Face `transformers` `pp_ocrv5_mobile_rec` (the rec class that
  `lang=hi` actually selects)

## What Table 1 already uses

`src/eval/run_baselines.py` `_get_paddle_ocr` / `run_paddleocr`:
`PaddleOCR(lang=...)` 3.x API, `predict()`, Hindi `lang='hi'`. Hindi-only
exact match **9.2%** lives in `docs/tier0e_paddleocr.md` / Follow-Up 7 —
cite those files; this page does not restate the taxonomy table.

That wrapper returns page `rec_texts` / `rec_scores` (mean line
confidence). It never opens the rec head.

## Which recognition model `lang=hi` loads

`paddleocr/_pipelines/ocr.py` `_get_ocr_model_names` (installed 3.7.0):

- `hi` ∈ `DEVANAGARI_LANGS` (`paddleocr/_utils/langs.py`)
- default `ppocr_version` for that set is **PP-OCRv5** (not v6; v6 is
  `ch` / `en` / `japan` / most Latin)
Confirmed by calling the installed helper (no weights download):

```
PaddleOCR._get_ocr_model_names(None, "hi", None)
# ('PP-OCRv5_server_det', 'devanagari_PP-OCRv5_mobile_rec')
```

That name is in `TEXT_REC_TRANSFORMERS_MODELS` in
`paddlex/inference/models/text_recognition/predictor.py`, so inference
goes through `TextRecTransformersPredictor`, not the Paddle-native CTC
`TextRecRunnerPredictor` path.

## Is there a generation loop with per-step logits?

**No.** Recognition is a **single forward** over the line crop:

1. `TextRecTransformersPredictor.process` (`predictor.py`): preprocess
   images → `self.forward(model_inputs)` → `postprocess`.
2. `TransformersPredictor.forward`
   (`paddlex/inference/models/predictors/transformers_predictor.py`)
   calls `self.infer(**model_inputs)` under `inference_mode`. A
   `generate()` helper exists on that base class; **text recognition
   does not call it**.
3. `PPOCRV5MobileRecForTextRecognition.forward`
   (`transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py`):
   backbone → rec head. The head is a linear projection to
   `config.head_out_channels` (default **18385**) then
   `F.softmax(..., dim=2)`. Output layout is
   `(batch, width_steps, vocab)`, i.e. **CTC-style time axis**, not
   autoregressive token steps.
4. Decoding is greedy CTC collapse:
   `PPOCRV5ServerRecImageProcessor.post_process_text_recognition`
   (shared pattern; docstring says logits / probability maps
   `(batch, height, vocab_size)` — the tensor is the width axis in the
   mobile rec head). Argmax over vocab per timestep, drop blanks
   (`index == 0`) and consecutive duplicates, join
   `character_list[id]`.

The public `PaddleOCR.predict()` result therefore has **no** sequence of
decoder steps comparable to `src/models/instrument/generate.py`.

Could one *hook* `last_hidden_state` after `forward`? Yes, that tensor
is the CTC softmax (already softmaxed in the head). That is **not**
teacher-forced log p of a grapheme string at Table 6 positions. Getting
a sequence likelihood would mean the **CTC forward algorithm** (sum
over alignments), which this repo’s probes do not implement and which
is a different estimand than the instrument’s teacher-forced
p(g_i | g_<i, image).

## Vocabulary vs grapheme clusters

Instrument tokenizer: Unicode grapheme clusters (`regex` `\X`, Decision
#2), ~hundreds of ids including matra-composed glyphs as one token.

Paddle rec: `CTCLabelDecode` / `character_list` — a **character
dictionary**. `BaseRecLabelDecode` builds `dict[char] = i` from that
list and `decode` does `"".join(character_list[id] ...)`. Default head
width **18385** is a codepoint/symbol table, not grapheme clusters.
Devanagari matras are typically separate dictionary entries; a visual
cluster like `कि` is **not** one id. Aligning CTC timesteps to
instrument positions would need a custom many-to-one map. That is the
same class of vocabulary-alignment problem that stopped a honest Surya
control, plus CTC collapse.

There is no tokenizer API that emits grapheme ids.

## Mapping the requested suite (and why each item fails)

Requested, matching the instrument / Table 6 / Table 5:

| Requested measurement | PaddleOCR 3.7 `devanagari_PP-OCRv5_mobile_rec` |
|---|---|
| Position-resolved teacher-forced log p(GT), Table 6 buckets | No AR teacher forcing. CTC time ≠ grapheme index 0, 1, 2–9, … |
| Rank of GT at position 0 (median, IQR vs instrument 70.5) | No “position 0 = first grapheme” distribution. Width-bin 0 is a CNN/SVTR column. |
| Self-generated max-softmax by position, text vs blank | Rec head softmax is over **width**, then CTC collapse to a string. `rec_scores` is mean kept-column max-prob (`post_process_text_recognition`), not per-grapheme max-softmax. Blank input is runnable at the pipeline level but the series is still not Table 6. |
| Encoder-zeroing + Table 5 agree/flip KL | No decoder prefix, no cross-attn over encoder memory in the instrument sense. Zeroing `feature_maps` before the rec head would be a **forked forward**, not `predict()`. KL between two CTC length-T softmaxes is not Decision #56’s KL on a shared greedy prefix. |

## Stop

Do not build a “Paddle diagnostic” that reports CTC column 0 as Table 6
position 0, or mean `rec_score` as calibration confidence, or a patched
`forward` as encoder ablation. Table 1 / `docs/tier0e_paddleocr.md`
already records that this engine **reads some Devanagari** (low exact
match). That is a corpus taxonomy fact, not a mechanistic positive
control.

If a later effort wants *any* Paddle number in this family, it has to
be specified as **CTC alignment likelihood** (or similar) and evaluated
as that, not as a drop-in for the instrument probes.

## Reproduce this inspection

```python
import paddleocr, inspect
from paddleocr._pipelines.ocr import PaddleOCR
print(paddleocr.__version__)
# _get_ocr_model_names is on the class; lang='hi' → devanagari_PP-OCRv5_mobile_rec
```

Installed files (this machine, 3.7.0): `paddleocr/_pipelines/ocr.py`,
`paddlex/inference/models/text_recognition/predictor.py`,
`transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py`.

# PaddleOCR positive control — Phase 1 feasibility

**Verdict: not viable.** The recognition model already used for Table 1
Hindi (`lang='hi'` in `src/eval/run_baselines.py`) is **CTC**, not an
autoregressive decoder. That single fact stops the instrument-matched
suite. No diagnostic code, no Colab, no CPU timing of a suite that
cannot be defined.

Inspected **installed** PaddleOCR **3.7.0** and the transformers rec
class it loads — not blog posts, not an older 2.x tree.

```
import paddleocr
from paddleocr._pipelines.ocr import PaddleOCR
paddleocr.__version__  # 3.7.0
PaddleOCR._get_ocr_model_names(None, "hi", None)
# ('PP-OCRv5_server_det', 'devanagari_PP-OCRv5_mobile_rec')
```

That rec name is in `TEXT_REC_TRANSFORMERS_MODELS`
(`paddlex/inference/models/text_recognition/predictor.py`), so
inference is `TextRecTransformersPredictor`.

Same stop rule as Surya (`docs/tier1_surya_control.md`): no forced
loads, no silent approximations, no treating a different estimand as
Table 6 / Table 5.

---

## Q1 — Autoregressive decoder or CTC?

**CTC.** Not an autoregressive / BOS-conditioned text decoder.

SVTR appears in this model, but as a **visual encoder** over the
backbone feature map, not as a token decoder you can teacher-force:

- `PPOCRV5MobileRecModel.forward` (`transformers/.../modeling_pp_ocrv5_mobile_rec.py`):
  backbone on `pixel_values`, then `avg_pool2d`. Output is a feature
  map, not a token sequence.
- `PPOCRV5MobileRecEncoderWithSVTR`: conv + `svtr_block` on flattened
  spatial tokens, reshape back to `(batch, C, H, W)`, then squeeze
  height and transpose to a **width** axis.
- `PPOCRV5MobileRecHead.forward`: `Linear(hidden, head_out_channels)`
  then `F.softmax(..., dim=2)`. Default `head_out_channels` is **18385**
  (`configuration_pp_ocrv5_mobile_rec.py`). Layout is
  `(batch, width_steps, charset)`.
- `PPOCRV5MobileRecForTextRecognition` has **no** `generate` method.
- Decode is CTC collapse: `CTCLabelDecode` prepends `"blank"` at index 0
  (`processors.py` `add_special_char`). HuggingFace
  `post_process_text_recognition` takes `argmax` over the last dim,
  drops index 0 and consecutive duplicates, joins `character_list[id]`.

There is no BOS, no `force_next_ids`, no p(g_i | g_<i, image) at
grapheme step i. Width-bin t is an image-column alignment, the thing
the prompt says does **not** transfer.

“PP-OCRv5 uses SVTR” does not make this model teacher-forcible like
the instrument. SVTR here is CRNN-family: visual sequence in, CTC
charset out.

**Stop.** Q2 and Q3 are recorded for completeness. They are not needed
to close Phase 1.

---

## Q2 — Can per-step softmax be taken from a generation loop?

**There is no generation loop** on the Table 1 rec path.

`TextRecTransformersPredictor.process` (`predictor.py`):

1. preprocess images
2. `self.forward(model_inputs)` → `self.infer(**model_inputs)` under
   `torch.inference_mode` (`transformers_predictor.py`)
3. `postprocess` → `image_processor.post_process_text_recognition`

`TransformersPredictor.generate` exists on the base class. **Text
recognition never calls it.** The rec `forward` is one shot.

One *could* read `last_hidden_state` after `forward` (already softmaxed
in the head). That is a CTC time-softmax, not decoder-step softmax.
Using it as Table 6 “position 0” would be a silent approximation. Not
done.

---

## Q3 — Tokenizer vs grapheme clusters (`regex` `\X`)?

**No workable 1–1 map**, same class of problem as Surya, plus CTC.

- Instrument (Decision #2): grapheme clusters, including matra-composed
  glyphs as one id.
- Paddle: `character_list` / `CTCLabelDecode.dict[char] = i`. Entries
  are charset symbols (codepoints), not `\X` clusters. `कि` is not one
  rec id.
- Head width 18385 is a large character table, not the instrument’s
  ~367 clusters.

Even a forced many-to-one map would still sit on CTC columns, not
teacher-forced grapheme positions.

---

## Phase 2

**Not started.** Q1 is CTC, so the requested suite (Table 6 buckets,
position-0 GT rank vs 70.5, max-softmax by grapheme position, blank
series, Table 5 encoder-zeroing on a shared greedy prefix) does not
transfer. `docs/paddleocr_positive_control.md` stays empty of those
numbers.

No CPU wall-clock: nothing to time that would answer those probes.

Table 1 / `docs/tier0e_paddleocr.md` already record that this engine
reads some Devanagari (low exact match). That is taxonomy, not a
mechanistic positive control.

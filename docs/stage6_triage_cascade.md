# Stage 6 — triage cascade

Depends on Stage 5b's Sarvam cache/jsonl (Decision #19: no new API).
Router quality, not cost savings (Decision #16).

## Protocol

- **Kept-page error:** grapheme CER of Sarvam Extract vs GT.
- **Escalated pages:** treated as CER = 0 (perfect human review).
- **System error at k:** mean CER over all n pages after filling
  zeros on the k escalated pages. Same k for every policy.
- **Instrument policy:** escalate the k *lowest* Probe 5b
  `mean_confidence` pages (3-seed mean, Hindi condition).
- **Random:** mean system error over 1000 draws of k pages
  (RNG seed 0).
- **Layout:** escalate the k *longest* GT grapheme strings.
  (Line crops have no Stage 3 layout-bucket labels.)
- **Tesseract:** escalate the k lowest Tesseract confidences
  on the matching `*_plain.png`.
- **Fair slice reported in the headline:** k = round(0.2 × n),
  plus the full k = 0…n table.

## Results

**Not computed.** Stage 5b ρ has not been authorised yet;
this cascade would peek at the same Sarvam CERs. Run after
Stage 5b `--compute-now`.

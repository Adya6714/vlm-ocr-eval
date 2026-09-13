# Tier 0d — noise and scrambled teacher-forcing

**Status: computed** on Colab T4. Producer:
`src/probes/probe_gt_likelihood.py --extra-conditions noise scrambled`
(Cell 6). Same 60-image Hindi pool as Table 6. Sibling jsonl so
committed real/blank `probe_gt_likelihood_hindi_natural_seed*.jsonl`
stay intact. Four conditions/seed because the script always scores
real+blank plus the extras (60×4=240 rows/seed).

Grayscale `make_matched_noise` (`image.convert("L")`) ran without
the RGB `fromarray` crash.

| seed | condition | n | mean log p(GT) | mean entropy |
|---|---|---:|---:|---:|
| 0 | blank | 60 | -1.5214 | 0.0349 |
| 0 | noise | 60 | -1.5161 | 0.0283 |
| 0 | real | 60 | -1.5915 | 0.0243 |
| 0 | scrambled | 60 | -1.6070 | 0.0259 |
| 1 | blank | 60 | -1.8389 | 0.0213 |
| 1 | noise | 60 | -1.8363 | 0.0257 |
| 1 | real | 60 | -1.8445 | 0.0169 |
| 1 | scrambled | 60 | -1.7706 | 0.0218 |
| 2 | blank | 60 | -1.8940 | 0.0196 |
| 2 | noise | 60 | -1.9147 | 0.0193 |
| 2 | real | 60 | -1.9130 | 0.0216 |
| 2 | scrambled | 60 | -1.8124 | 0.0254 |

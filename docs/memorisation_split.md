# Memorisation split (Section 9 AUROC)

Question: does the synthetic-natural AUROC in
`docs/paper_defensibility_stats.md` (pooled **0.8381**, cite that
file — do not treat this page as a second copy of Follow-Up 3)
rank genuine correctness, or only whether the line was in training?

## Method

- Training strings: exact `text` field of
  `data/manifests/hindi_natural.jsonl` (n=2538 rows, 2348 unique `text` values).
- Eval: `records[]` in committed
  `data/probe_results/probe5_hindi_natural_seed{0,1,2}.jsonl`.
- Match rule: `ground_truth ==` some training `text` (verbatim;
  no NFC extra pass, no grapheme rewrite).
- Metrics: line accuracy (`correct` already uses Tier 1/2 in Probe 5)
  and Mann–Whitney AUROC (`paper_defensibility_stats.auroc`) on
  `(confidence, correct)`. Same estimator as Follow-Up 3.
- No new model forwards. Producer of the jsonl:
  `src/probes/probe5_calibration.py` samples
  `random.Random(0).sample(manifest_rows, n_samples)` from this
  same manifest, then scores the trained instrument.

| Seed | subset | n | accuracy | AUROC |
|---|---|---:|---:|---:|
| 0 | all | 100 | 0.1700 | 0.7633 |
| 0 | in training manifest | 100 | 0.1700 | 0.7633 |
| 0 | not in training manifest | 0 | n/a | n/a |
| 1 | all | 100 | 0.1400 | 0.9244 |
| 1 | in training manifest | 100 | 0.1400 | 0.9244 |
| 1 | not in training manifest | 0 | n/a | n/a |
| 2 | all | 100 | 0.2400 | 0.8317 |
| 2 | in training manifest | 100 | 0.2400 | 0.8317 |
| 2 | not in training manifest | 0 | n/a | n/a |

### Pooled (seeds 0–2 concatenated)

| subset | n | accuracy | AUROC |
|---|---:|---:|---:|
| all | 300 | 0.1833 | 0.8381 |
| in training manifest | 300 | 0.1833 | 0.8381 |
| not in training manifest | 0 | n/a | n/a |

## Finding

**The non-matching subset is empty (n=0).** Every Probe 5
synthetic-natural evaluation line (n=300 rows across three
seeds; 98 unique `ground_truth` strings) appears
verbatim in `hindi_natural.jsonl`. Probe 5 samples with
`random.Random(0).sample` from that same file
(`src/probes/probe5_calibration.py` `run_probe5`), so overlap is
100% by construction. n=0 is not a small held-out slice: there
is no held-out slice. AUROC on the non-matching group is not
defined and is not reported as 0.5 or as a collapse.

Pooled AUROC on the matching (only) subset is the same
computation as Follow-Up 3: 0.8381 (accuracy 0.1833).
Section 9's 0.838 therefore cannot be read as confidence tracking
correctness on unseen text.

Settling that claim needs a Probe 5 eval whose GT strings are
disjoint from the training `text` set. That rerun is not this
script.

## Not this eval: 60-image GT-likelihood pool

Teacher-forced Table 6 uses real scans, not Probe 5 crops. Unique
GT strings in `data/probe_results/probe_gt_likelihood_hindi_natural_seed0.jsonl` condition=real:
60 unique / 60 rows;
**19** of those unique strings appear verbatim in
`hindi_natural.jsonl`. That overlap is a contamination note for
Section 8 / n-gram splits, not a substitute for the Probe 5 AUROC
split above.

## Reproduce

```
PYTHONPATH=src python src/analysis/memorisation_vs_correctness.py
```

Output: `docs/memorisation_split.md`.


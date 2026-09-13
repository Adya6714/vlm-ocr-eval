# Tier 0d — noise and scrambled teacher-forcing

**Status: not computed this session.** The grayscale fix is in the tree.
The T4 re-run that would confirm it is not.

## Grayscale fix (code, not a result)

`src/probes/probe3_blank_control.py` `make_matched_noise` converts with
`image.convert("L")` before `np.array(...)`, then `Image.fromarray(...,
mode="L")`. That path is what `src/probes/probe_gt_likelihood.py`
`render_condition_image` uses for `condition=noise`. RGB crops used to
raise `Too many dimensions: 3 > 2`. **Seeing that function in git is
not a substitute for Cell 6 numbers.** This session did not execute
`probe_gt_likelihood.py --extra-conditions noise scrambled`.

## What was checked on this machine (2026-09-13)

| Check | Result |
|---|---|
| `data/probe_results/probe_gt_likelihood_extra_hindi_natural_seed{0,1,2}.jsonl` | **absent** |
| Hindi instrument checkpoints (`checkpoint_hindi_natural_seed*.pt`) | **absent** (`docs/tier0_checkpoint_status.md`) |
| CUDA T4 | **not this laptop** |
| Committed `probe_gt_likelihood_hindi_natural_seed*.jsonl` | still **real + blank only** |

`src/colab/notebook_support.py` `write_tier0d` would mark the extra
files **Incomplete** if Cell 6 were invoked here. It was not.

## What Cell 6 is supposed to do

Notebook: `notebooks/colab_run.ipynb` Cell 6. Same 60-image Hindi pool
as Table 6 (`build_hindi_sample` / `n-samples 100` on a 60-image set).
Sibling jsonl so committed real/blank files stay untouched:

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_gt_likelihood.py \
    --script hindi --condition natural --seed $s \
    --output-root "$INSTRUMENT_CKPT" --data-root data \
    --n-samples 100 --device cuda \
    --extra-conditions noise scrambled \
    --out data/probe_results/probe_gt_likelihood_extra_hindi_natural_seed${s}.jsonl
done
PYTHONPATH=src python -c "from pathlib import Path; from colab.notebook_support import write_tier0d; write_tier0d(Path('.'))"
```

Resume key is `(condition, image_path)`. After a real run, overwrite
this file with the `write_tier0d` table (mean log p(GT) and entropy by
seed × condition) and a Follow-Up 6 bucket breakdown if the jsonl has
`step_log_p_gt`. Do not paste real/blank Table 6 numbers here as if they
were noise/scrambled.

## Failure mode if the fix were insufficient

Cell 6 would raise on `make_matched_noise` / `fromarray` for RGB crops
and `write_tier0d` would still see missing jsonl. That traceback has
**not** been observed this session because the probe did not start.

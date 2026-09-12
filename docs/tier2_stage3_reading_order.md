# Tier 2 / Stage 3 — reading-order and table-binding metrics

This step **does not** need the demo LoRA. Metrics score any predicted
block permutation / cell-to-header assignment against GT.

## What ran

Unit tests (this session, `python3 -m unittest tests.test_structure_metrics
tests.test_demo_rlvr_order`): **18/18 OK**.

Layout-bank inventory (this session):

```text
python3 src/eval/reading_order_metric.py
```

on `data/cache/layouts/bank.json`:

| Category | n templates | Geometric baseline mean Kendall tau |
|---|---:|---:|
| single-column | 25 | 0.733 |
| two-column | 2 | 0.600 |
| marginalia | 1 | 1.0 |
| table-embedded | 0 | (empty; `mean_tau` is `None`) |
| form | 0 | (empty) |

Other bank facts measured from the same file:

- **28** templates total.
- **2** regions with `kind=table` (tiny Wikipedia infobox/navbox boxes).
- **0** regions that meet `layout_sources.classify_layout`’s “real table”
  size cut (`height ≥ 0.18` and `width ≥ 0.25`).
- **0** `form_field` regions.

`render.py` paints only `text` / `header` / `footer` / `margin`. Table
and form regions are **not** drawn and have **no cell-level GT**.

## Design (matches BOOK.md Chapter 5)

- **Reading order is a permutation, not classification.**
  `kendall_tau_order` requires the same id set; dropping a block is a
  hard error (coverage belongs in RLVR, not here). Identity → 1.0;
  full reversal (n≥2) → −1.0; adjacent swap sits between.
- **Curve, not one number.** `tau_by_bucket` always emits the five
  complexity labels. Empty buckets stay `n=0`, `mean_tau=None` so a
  plot can show a hole instead of averaging them away.
- **Geometric baseline** is top-to-bottom then left-to-right on region
  boxes. The 0.733 single-column tau already shows that even “simple”
  Wikipedia region order is not identical to a y-then-x sort (headers,
  footers, infobox fragments). With n=2 two-column pages, the 0.600
  mean is **not** a stable complexity curve.
- **Table binding (Decision #12):** max horizontal overlap of a cell
  box onto header boxes. Tests use a synthetic aligned grid (geometry
  scores 1.0 there by construction). That is a fixture, **not** a
  renderer result.

## Layout bank vs new synthetic data

**Do not generate a fake table-embedded bank to fill the curve.** The
live bank already has region-level `reading_order` on every template;
that is enough to score **block order** for single-column / two-column /
marginalia. It is **not** enough for table-binding or the form bucket.

To finish Stage 3 acceptance (tau-vs-complexity **and** a binding
number on demo output):

1. Populate `table-embedded` and `form` from real PDFs (india.gov or
   equivalent), with cell boxes in GT.
2. Teach `render.py` to paint those cells (or score against PDF-extracted
   cell boxes without going through the text painter).
3. Run the metrics on a model’s predicted order — demo, instrument, or
   the geometric baseline — once that GT exists.

Until then, the geometric numbers above are a **baseline on region
templates**, labelled as such, not a demo-model result.

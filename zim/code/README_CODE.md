# Code

- `scripts/box_prior/` — Box-Support Prior implementation and its full evaluation
  (`box_prior_pipeline.py`), plus the staged helpers `00_init` … `04_tests`,
  a quick parameter probe (`quick_probe.py`) and the Composer demo (`composer_demo.py`).
- `scripts/final_claim/` — the metric pipeline: dataset/protocol preparation,
  metric computation, statistics, ablation and failure analysis. `final_claim_pipeline.py`
  holds the logic; `00`…`11` are thin stage entry points.
- `scripts/official_metrics.py` — SAD / MSE / MAE / boundary metric helpers.

Model weights, datasets and generated alpha outputs are not included.

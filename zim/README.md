# PromptMatte (ZIM-based)

Inference-only refinement on top of ZIM ViT-B box-prompt matting, evaluated on the
MicroMat official-prompt valid subset (706 images).

## Method

PromptMatte-TTA-GF + BSP

- **TTA-GF**: ZIM ViT-B with the official bbox prompt, horizontal-flip test-time
  augmentation, then a guided filter (r = 1) for boundary refinement.

  ```
  alpha0     = ZIM(I, bbox)
  alpha_flip = unflip(ZIM(flip(I), flip(bbox)))
  alpha_tta  = mean(alpha0, alpha_flip)
  alpha      = guided_filter(alpha_tta, I)
  ```

- **BSP** (Box-Support Prior): a box-derived spatial support prior that needs no ground
  truth, giving a small but consistent extra gain.

## Results (valid706)

| Method | SAD | MSE | MAE×1000 | Boundary SAD |
| --- | --- | --- | --- | --- |
| ZIM bbox (baseline) | 2.3201 | 0.000580 | 0.8255 | 0.7366 |
| + flip TTA | 2.1396 | 0.000453 | 0.7588 | 0.6692 |
| + guided filter (r1) | 2.2978 | 0.000568 | 0.8176 | 0.6990 |
| PromptMatte-TTA-GF | 2.1238 | 0.000446 | 0.7533 | 0.6386 |
| PromptMatte-TTA-GF + BSP | **2.1227** | **0.000445** | **0.7528** | **0.6383** |

Relative to the reproduced ZIM bbox baseline, PromptMatte-TTA-GF lowers SAD by ~8.5%,
MSE by ~23% and boundary SAD by ~13%; BSP adds a further small gain. Full numbers and
the holdout506 split are in `tables/`.

## Layout

- `code/scripts/box_prior/` — Box-Support Prior and its evaluation
- `code/scripts/final_claim/` — metric computation, statistics and reporting pipeline
- `code/scripts/official_metrics.py` — metric helpers
- `tables/` — leaderboards and ablation tables (CSV)
- `visuals/examples/` — representative comparison cases
- `visuals/failure_cases/` — failure examples
- `ppt_assets/method_diagram.png` — method diagram

Model weights, datasets and full alpha outputs are not included.

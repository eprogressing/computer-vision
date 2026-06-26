# Final Submission Summary

Generated at: `2026-06-04 16:19:23`

## 1. Dataset and protocol

All benchmark numbers in this package use the MicroMat official-prompt valid706 subset. This is not full MicroMat-3K. The fixed final holdout table uses the 506-row `final_holdout506` manifest from the BSP run.

## 2. Baselines

The primary baseline is ZIM ViT-B with official bbox prompt. Additional fair components include horizontal flip TTA and guided filtering.

| method | display_method | n_ok | SAD | MSE | MAE_x1000 | Boundary_SAD | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zim_vitb_bbox_default | ZIM bbox baseline | 706.000000 | 2.320107 | 0.000579536 | 0.825480 | 0.736623 | 0.000000 |
| zim_vitb_flip_tta_bbox | ZIM bbox + horizontal flip TTA | 706.000000 | 2.139645 | 0.000453471 | 0.758781 | 0.669200 | 0.000000 |
| zim_vitb_bbox_default_guided_r1 | ZIM bbox + guided filter r1 | 706.000000 | 2.297811 | 0.000568166 | 0.817642 | 0.698974 | 0.000000 |
| zim_vitb_flip_tta_bbox_guided_r1 | PromptMatte-TTA-GF | 706.000000 | 2.123826 | 0.000445816 | 0.753268 | 0.638556 | 0.000000 |
| promptmatte_tta_gf_box_support_prior | PromptMatte-TTA-GF+BSP | 706.000000 | 2.122658 | 0.000445414 | 0.752814 | 0.638283 | 0.000000 |


## 3. PromptMatte-TTA-GF

PromptMatte-TTA-GF = ZIM ViT-B official bbox prompt + horizontal flip TTA + guided filter r1. It is the stable main backend before BSP.

## 4. Box-Support Prior

BSP = Box-Support Prior. It uses the official bbox as a no-GT spatial support prior to conservatively suppress alpha leakage outside the prompt support.

## 5. Results

Core valid706 comparison:

| method | display_method | n_ok | SAD | MSE | MAE_x1000 | Boundary_SAD |
| --- | --- | --- | --- | --- | --- | --- |
| zim_vitb_bbox_default | ZIM bbox baseline | 706.000000 | 2.320107 | 0.000579536 | 0.825480 | 0.736623 |
| zim_vitb_flip_tta_bbox_guided_r1 | PromptMatte-TTA-GF | 706.000000 | 2.123826 | 0.000445816 | 0.753268 | 0.638556 |
| promptmatte_tta_gf_box_support_prior | PromptMatte-TTA-GF+BSP | 706.000000 | 2.122658 | 0.000445414 | 0.752814 | 0.638283 |


Final holdout comparison:

| method | display_method | n_ok | SAD | MSE | MAE_x1000 | Boundary_SAD |
| --- | --- | --- | --- | --- | --- | --- |
| zim_vitb_bbox_default | ZIM bbox baseline | 506.000000 | 2.324162 | 0.000591247 | 0.834215 | 0.728160 |
| zim_vitb_flip_tta_bbox | ZIM bbox + horizontal flip TTA | 506.000000 | 2.116439 | 0.000447704 | 0.757371 | 0.659688 |
| zim_vitb_bbox_default_guided_r1 | ZIM bbox + guided filter r1 | 506.000000 | 2.302180 | 0.000580030 | 0.826511 | 0.690886 |
| zim_vitb_flip_tta_bbox_guided_r1 | PromptMatte-TTA-GF | 506.000000 | 2.100985 | 0.000440252 | 0.752010 | 0.629427 |
| promptmatte_tta_gf_box_support_prior | PromptMatte-TTA-GF+BSP | 506.000000 | 2.100456 | 0.000440099 | 0.751811 | 0.629145 |


## 6. Ablation

The ablation table is exported to `ppt_assets/03_ablation_table.png` and `tables/ablation_table.csv`. Failed MS-TTA/risk/LHR/TTA-GF++ explorations are not used as final methods.

Selected improvement rows:

| comparison | metric | right_value | left_value | relative_improvement_percent |
| --- | --- | --- | --- | --- |
| PromptMatte-TTA-GF vs ZIM bbox baseline | SAD | 2.320107 | 2.123826 | 8.459977 |
| PromptMatte-TTA-GF vs ZIM bbox baseline | MSE | 0.000579536 | 0.000445816 | 23.073725 |
| PromptMatte-TTA-GF vs ZIM bbox baseline | MAE_x1000 | 0.825480 | 0.753268 | 8.747862 |
| PromptMatte-TTA-GF vs ZIM bbox baseline | Boundary_SAD | 0.736623 | 0.638556 | 13.313017 |
| PromptMatte-TTA-GF+BSP vs ZIM bbox baseline | SAD | 2.320107 | 2.122658 | 8.510322 |
| PromptMatte-TTA-GF+BSP vs ZIM bbox baseline | MSE | 0.000579536 | 0.000445414 | 23.143052 |
| PromptMatte-TTA-GF+BSP vs ZIM bbox baseline | MAE_x1000 | 0.825480 | 0.752814 | 8.802911 |
| PromptMatte-TTA-GF+BSP vs ZIM bbox baseline | Boundary_SAD | 0.736623 | 0.638283 | 13.350103 |
| BSP increment over PromptMatte-TTA-GF | SAD | 2.123826 | 2.122658 | 0.054998 |
| BSP increment over PromptMatte-TTA-GF | MSE | 0.000445816 | 0.000445414 | 0.090122 |
| BSP increment over PromptMatte-TTA-GF | MAE_x1000 | 0.753268 | 0.752814 | 0.060325 |
| BSP increment over PromptMatte-TTA-GF | Boundary_SAD | 0.638556 | 0.638283 | 0.042782 |


## 7. Composer Showcase

Composer assets are application-only outputs: RGBA, background replacement, blur background, and alpha-edge visualization. They are copied/linked under `composer/` and `ppt_assets/06_composer_showcase/`.

## 8. SAM2 teammate integration

No aligned SAM2 teammate result was found. See `sam2_integration/SAM2_RESULT_MISSING_TODO.md` and the desktop teammate package for exact required files and schemas.

## 9. Failure and limitations

- valid706 subset only.
- BSP gives a small but consistent incremental improvement over PromptMatte-TTA-GF.
- Composer is not a benchmark method.
- SAM2 result is pending teammate submission.

## 10. Claim safety

Allowed and forbidden claims are written in `claim_safety/FINAL_ALLOWED_CLAIMS.md` and `claim_safety/FINAL_FORBIDDEN_CLAIMS.md`.

## 11. Reproduction

Run `bash repro/commands_reproduce_final_package.sh` from this package, or run `python3 scripts/final_submission/final_submission_package.py` from `/home/lpy/anisorisk/computer_vison`.

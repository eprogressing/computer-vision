# PromptMatte-TTA-GF Final Summary

## 1. TL;DR

- final method: `PromptMatte-TTA-GF` = `zim_vitb_flip_tta_bbox_guided_r1`
- baseline: `zim_vitb_bbox_default`
- valid subset count: `706`
- course submission verdict: `PASS`
- paper-level claim verdict: `NO-GO`
- SAD improvement over baseline: `0.08459976985271576`
- MSE improvement over baseline: `0.2307372451605895`
- MAE_x1000 improvement over baseline: `0.08747862453989781`
- Boundary_SAD improvement over baseline: `0.13313016936341096`
- official Grad/Conn status: `bounded sanity check only; full split approximate`
- failed adaptive ablation: `risk_gated_adaptive_refinement` is not the main method

## 2. Why The Method Was Simplified

Previous selector/adaptive variants did not produce a stronger final method. The stable and interpretable method is ZIM bbox inference with flip-TTA and guided filtering.

## 3. Dataset Protocol

- audited rows: `3000`
- repaired valid official prompt rows: `706`
- wording: local MicroMat official-prompt valid subset; not full MicroMat-3K unless all rows are valid.

## 4. Method

alpha_0 = ZIM(I, bbox)

alpha_flip = unflip(ZIM(flip(I), flip(bbox)))

alpha_tta = average(alpha_0, alpha_flip)

alpha_final = guided_filter_boundary(alpha_tta, I)

## 5. Main Results

| method | display_method | ok | SAD | MSE | MAE_x1000 | Boundary_SAD | Gradient_official_zim | Connectivity_official_zim | Gradient_approx | Connectivity_approx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zim_vitb_flip_tta_bbox_guided_r1 | PromptMatte-TTA-GF | 706 | 2.123826205445501 | 0.00044581581196763713 | 0.7532680459431355 | 0.6385563802047931 | 1.1191785267631833 | 1.4704120114445687 | 0.415287072482722 | 1.9612067988668556 |
| zim_vitb_flip_tta_bbox | zim_vitb_flip_tta_bbox | 706 | 2.139644727490779 | 0.0004534709738699304 | 0.7587811886187822 | 0.6691997890748485 | nan | nan | 0.4613386263632909 | 1.958032577903683 |
| risk_gated_adaptive_refinement | risk_gated_adaptive_refinement | 706 | 2.18041854178686 | 0.0004826553000170745 | 0.7745563837517548 | 0.6482386623559356 | nan | nan | 0.4544086380684933 | 2.098790368271955 |
| zim_vitb_bbox_default_guided_r1 | zim_vitb_bbox_default_guided_r1 | 706 | 2.297810662759431 | 0.0005681659373770037 | 0.8176421160907313 | 0.6989738599373497 | nan | nan | 0.45012410185576834 | 2.1996501416430596 |
| zim_vitb_bbox_default | zim_vitb_bbox_default | 706 | 2.3201066981420637 | 0.0005795364576837008 | 0.8254798914308504 | 0.7366231441413377 | 1.2688404337192576 | 0.9228237122297287 | 0.510187134452734 | 2.1985509915014165 |

## 6. Ablation

See `FAIR_BASELINE_AND_ABLATION_REPORT.md`.

## 7. Statistics

See `FINAL_STATISTICAL_REPORT.md`.

## 8. Failure Cases

See `FINAL_FAILURE_ANALYSIS_REPORT.md`.

## 9. Visuals And PPT Assets

See `visuals/index.html` and `ppt_assets/`.

## 10. Claim Safety

Allowed: local official-prompt subset improvement over reproduced ZIM bbox baseline; no-GT inference; simple TTA + guided filtering module.

Forbidden: paper-level SOTA; full MicroMat-3K; official Grad/Conn if only approximate; selector main contribution; text benchmark fixed.

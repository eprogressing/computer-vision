# SAM2 Result Missing TODO

No aligned SAM2 teammate benchmark result is available in this final package.

Latest auto-detected SAM-related candidate:

`/home/lpy/anisorisk/computer_vison/runs/text_branch_gsam2_20260601_085004`

This candidate is not accepted as teammate benchmark data because it does not provide a 706-row aligned result table over the valid706 manifest. The detected Grounded-SAM2 text branch is blocked before actual text inference and cannot be used as SAM2 benchmark evidence.

## Files the teammate must provide

1. `metrics_all.csv`
   - Per-sample rows.
   - Required columns: `sample_id,method,status,SAD,MSE,MAE_x1000,Boundary_SAD,alpha_path_pred`.
   - Must use the exact `sample_id` values from `valid706_manifest_for_alignment.csv`.
2. `leaderboard.csv`
   - Required columns: `method,n,ok,failure_rate,SAD,MSE,MAE_x1000,Boundary_SAD`.
   - Include rows such as `sam2_bbox_binary` and/or `sam2_guided`.
3. `outputs/{method}/{sample_id}/alpha.png`
   - One alpha image per method/sample.
4. `manifest.csv`
   - The manifest actually used for inference.
5. `method_notes.md`
   - Prompt type, checkpoint, post-processing, and whether any samples failed.

## Alignment rules

- Do not use GT alpha for inference.
- Use the official bbox prompt protocol where applicable.
- If `n_ok != 706`, provide per-method `n_ok` and the missing/failed `sample_id` list.
- Composer/demo images must not be mixed into benchmark metrics.
- Approximate Grad/Conn, if any, must not be labeled official.

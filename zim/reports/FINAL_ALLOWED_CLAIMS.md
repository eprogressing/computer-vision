# Final Allowed Claims

- Dataset/protocol: MicroMat official-prompt valid706 subset, not full MicroMat-3K.
- Baseline: reproduced ZIM ViT-B official bbox prompt baseline on the same valid706 subset.
- Main method: PromptMatte-TTA-GF+BSP is an inference-only backend refinement.
- PromptMatte-TTA-GF means ZIM ViT-B official bbox prompt + horizontal flip TTA + guided filter r1.
- BSP means Box-Support Prior: a no-GT spatial support prior derived from the official bbox prompt.
- Metrics reported in the final leaderboard are local SAD, MSE, MAE_x1000, and Boundary_SAD on fixed sample sets.
- Composer is an application showcase for RGBA, background replacement, blur background, and alpha-edge visualization.

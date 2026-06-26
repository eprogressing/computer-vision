#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["05_consolidate_fair_baselines_and_ablation", *(__import__("sys").argv[1:])])

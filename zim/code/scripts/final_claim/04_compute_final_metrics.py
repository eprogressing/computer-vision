#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["04_compute_final_metrics", *(__import__("sys").argv[1:])])

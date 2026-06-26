#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["03_prepare_metric_runtime", *(__import__("sys").argv[1:])])

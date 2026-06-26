#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["07_final_failure_analysis", *(__import__("sys").argv[1:])])

#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["10_run_unit_tests", *(__import__("sys").argv[1:])])

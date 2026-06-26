#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["11_make_final_report", *(__import__("sys").argv[1:])])

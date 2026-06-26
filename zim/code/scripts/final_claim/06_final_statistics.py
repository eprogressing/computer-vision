#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["06_final_statistics", *(__import__("sys").argv[1:])])

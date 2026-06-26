#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["01_ingest_and_freeze_final_method", *(__import__("sys").argv[1:])])

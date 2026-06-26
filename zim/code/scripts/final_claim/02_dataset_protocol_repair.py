#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["02_dataset_protocol_repair", *(__import__("sys").argv[1:])])

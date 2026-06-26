#!/usr/bin/env python
from final_claim_pipeline import main

if __name__ == "__main__":
    main(["08_final_protocol_audit", *(__import__("sys").argv[1:])])

#!/usr/bin/env python
from box_prior_pipeline import main

if __name__ == "__main__":
    main(["00_init", *(__import__("sys").argv[1:])])

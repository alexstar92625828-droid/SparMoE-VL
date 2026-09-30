#!/usr/bin/env python3
"""MoPE visual recovery Stage 1 on all 500,000 main-experiment pairs."""

from sparmoe_vl.baselines.vision.mope_recovery import run


if __name__ == "__main__":
    run(stage=1)

#!/usr/bin/env python3
"""MoPE Stage 2, replaying exactly the Stage-1 main-experiment data sequence."""

from sparmoe_vl.baselines.text.mope_recovery import run


if __name__ == "__main__":
    run(stage=2)

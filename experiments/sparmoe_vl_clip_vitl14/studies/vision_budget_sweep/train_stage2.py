"""Freeze one budget point's Stage-1 structure and train token routing."""

import sys

from sparmoe_vl.studies.vision_budget_sweep import run_training


if __name__ == "__main__":
    run_training(2, sys.argv[1:])

"""Learn SPG structure under one registered visual global budget."""

import sys

from sparmoe_vl.studies.vision_budget_sweep import run_training


if __name__ == "__main__":
    run_training(1, sys.argv[1:])

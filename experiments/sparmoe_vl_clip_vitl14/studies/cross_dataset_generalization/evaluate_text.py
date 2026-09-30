"""Evaluate the p=0.6 text main checkpoints for Table 4."""

import sys

from sparmoe_vl.studies.generalization.evaluate import main


if __name__ == "__main__":
    main(sys.argv[1:], modality="text")

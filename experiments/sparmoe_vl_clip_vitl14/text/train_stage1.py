"""Text Stage 1: learn SPG structure under global budget P."""

import sys

from sparmoe_vl.train_two_stage import main


if __name__ == "__main__":
    main(["--modality", "text", "--stage", "1", *sys.argv[1:]])

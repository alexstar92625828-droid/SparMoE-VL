"""Vision Stage 2: freeze Stage-1 structure and train token routers only."""

import sys

from sparmoe_vl.train_two_stage import main


if __name__ == "__main__":
    main(["--modality", "vision", "--stage", "2", *sys.argv[1:]])

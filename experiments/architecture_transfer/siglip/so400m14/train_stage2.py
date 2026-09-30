#!/usr/bin/env python3
"""Freeze SigLIP SO400M/14 Stage-1 structure and train token routers only."""

from sparmoe_vl.architecture_transfer.siglip.training import main_stage2


if __name__ == "__main__":
    main_stage2("so400m14")

#!/usr/bin/env python3
"""Aggregate three SigLIP ViT-SO400M/14 architecture-transfer seeds."""

from sparmoe_vl.architecture_transfer.siglip.summary import main


if __name__ == "__main__":
    main("so400m14")

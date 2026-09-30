#!/usr/bin/env python3
"""Evaluate one SigLIP ViT-SO400M/14 architecture-transfer seed."""

from sparmoe_vl.architecture_transfer.siglip.evaluation import main


if __name__ == "__main__":
    main("so400m14")

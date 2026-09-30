#!/usr/bin/env python3
"""Evaluate one SigLIP ViT-B/16 architecture-transfer seed."""

from sparmoe_vl.architecture_transfer.siglip.evaluation import main


if __name__ == "__main__":
    main("vit_b16")

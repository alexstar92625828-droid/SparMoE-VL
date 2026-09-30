#!/usr/bin/env python3
"""Evaluate one SigLIP2 ViT-B/16 architecture-transfer seed."""

from sparmoe_vl.architecture_transfer.siglip2.evaluation import main


if __name__ == "__main__":
    main("vit_b16")

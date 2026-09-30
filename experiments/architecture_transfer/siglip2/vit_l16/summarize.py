#!/usr/bin/env python3
"""Aggregate three SigLIP2 ViT-L/16 architecture-transfer seeds."""

from sparmoe_vl.architecture_transfer.siglip2.summary import main


if __name__ == "__main__":
    main("vit_l16")

#!/usr/bin/env python3
"""Freeze SigLIP2 ViT-B/16 Stage-1 structure and train token routers only."""

from sparmoe_vl.architecture_transfer.siglip2.training import main_stage2


if __name__ == "__main__":
    main_stage2("vit_b16")

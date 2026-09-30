#!/usr/bin/env python3
"""Freeze CLIP ViT-B/32 Stage-1 structure and train token routers only."""

from sparmoe_vl.architecture_transfer.clip.training import main_stage2


if __name__ == "__main__":
    main_stage2("vit_b32")

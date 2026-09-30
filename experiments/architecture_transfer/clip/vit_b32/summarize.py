#!/usr/bin/env python3
"""Aggregate three CLIP ViT-B/32 architecture-transfer seeds."""

from sparmoe_vl.architecture_transfer.clip.summary import main


if __name__ == "__main__":
    main("vit_b32")

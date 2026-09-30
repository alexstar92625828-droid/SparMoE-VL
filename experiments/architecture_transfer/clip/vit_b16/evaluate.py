#!/usr/bin/env python3
"""Evaluate one CLIP ViT-B/16 vision or text architecture-transfer seed."""

from sparmoe_vl.architecture_transfer.clip.evaluation import main


if __name__ == "__main__":
    main("vit_b16")

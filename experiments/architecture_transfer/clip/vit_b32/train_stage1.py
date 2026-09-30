#!/usr/bin/env python3
"""Learn CLIP ViT-B/32 SPG structure under the global budget."""

from sparmoe_vl.architecture_transfer.clip.training import main_stage1


if __name__ == "__main__":
    main_stage1("vit_b32")

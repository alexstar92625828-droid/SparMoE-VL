#!/usr/bin/env python3
"""Learn SigLIP ViT-B/16 SPG structure under the global budget."""

from sparmoe_vl.architecture_transfer.siglip.training import main_stage1


if __name__ == "__main__":
    main_stage1("vit_b16")

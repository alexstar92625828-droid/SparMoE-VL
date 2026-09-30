"""SigLIP2 bindings for the shared no-CLS/last-position SigLIP-family model.

SigLIP and SigLIP2 expose the same OpenCLIP tower layout used by this study:
the timm vision trunk routes every patch and the text tower keeps its last
pooled position dense. The implementation therefore lives once in the shared
SigLIP-family module; this file supplies explicit SigLIP2-facing names.
"""

from ..siglip.model import (
    NestedFFN,
    SigLIPSparMoE,
    build_model,
    controller_state,
    initialize_stage2,
    load_stage2_controller,
)


SigLIP2SparMoE = SigLIPSparMoE

__all__ = [
    "NestedFFN",
    "SigLIP2SparMoE",
    "build_model",
    "controller_state",
    "initialize_stage2",
    "load_stage2_controller",
]

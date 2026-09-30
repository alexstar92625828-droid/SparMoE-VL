"""CLIP ViT-B architecture-transfer experiments from paper Table 6."""

from .protocol import MODEL_SPECS, CLIPSpec, get_spec

__all__ = ["CLIPSpec", "MODEL_SPECS", "get_spec"]

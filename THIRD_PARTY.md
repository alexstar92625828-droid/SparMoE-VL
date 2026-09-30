# Third-party methods and assets

The comparison implementations are CLIP-specific adaptations of published
methods. Their upstream method references are recorded here and as immutable
identifiers in the corresponding Python modules.

| Method | Upstream reference | Recorded implementation revision |
|---|---|---|
| TEAL | [repository](https://github.com/FasterDecoding/TEAL) | `fb7373c93ac3594817c9ee64d4e08b47430a1822` |
| OPTIN | [paper](https://openreview.net/forum?id=MVmT6uQ3cQ); [repository](https://github.com/Skhaki18/optin-transformer-pruning) | `6c8d7caef2193a48c766f31e56008992dfc0c3bd` |
| FLAP | [paper](https://arxiv.org/abs/2312.11983); [repository](https://github.com/CASIA-LMC-Lab/FLAP) | `3bb57db3449dd2fa04a5c2192de80e87e33be2b1` |
| MoPE | [method reference](https://arxiv.org/abs/2403.07839) | adapted implementation |

These references identify the baseline methods and source materials; they do
not imply endorsement of SparMoE-VL by the upstream maintainers. Acknowledge
the original method when using or discussing a baseline.

SparMoE-VL also depends on separately obtained pretrained models and datasets,
including OpenCLIP/CLIP, SigLIP, SigLIP2, LLaVA, ShareGPT4V, COCO, Flickr30k,
CIFAR-100, ImageNet, Food-101, POPE, MME, GQA, and VQAv2. They are not
redistributed here and remain governed by their publishers' licenses and
terms.

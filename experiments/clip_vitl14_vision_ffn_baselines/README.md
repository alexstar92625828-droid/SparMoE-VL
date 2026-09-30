# CLIP ViT-L/14 vision FFN baselines

This suite contains the visual-encoder comparison methods used with
SparMoE-VL. Each method has an independent experiment directory; shared
visual data, model, and metric code lives in `sparmoe_vl.baselines.vision`.

Every method must consume the exact 500,000-image ShareGPT4V pool used by the
two-stage visual main experiment. Release artifacts are rejected unless they
record data seed 42 and ordered pool SHA-256
`f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0`.

| Method | Directory | Status |
|---|---|---|
| TEAL-CLIP-FFN | `teal/` | ready |
| OPTIN-CLIP-FFN | `optin/` | ready |
| FLAP-CLIP-FFN | `flap/` | ready |
| MoPE-CLIP-FFN | `mope/` | ready |

Generated thresholds and evaluation summaries belong under
`outputs/clip_vitl14_vision_ffn_baselines/` and are not stored in this code
repository.

Method references, upstream repositories, and recorded revisions are listed in
the repository-level `THIRD_PARTY.md`.

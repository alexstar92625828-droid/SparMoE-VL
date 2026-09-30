# CLIP ViT-L/14 text FFN baselines

This experiment suite contains the text-encoder comparison methods used with
SparMoE-VL. Each method owns an independent directory, while shared dataset,
model-loading, retrieval, and metric code lives in
`sparmoe_vl.baselines.text`.

All methods consume the exact 500,000-example ShareGPT4V text pool used by the
two-stage SparMoE-VL text experiment. A method must validate data seed 42 and
the ordered pool SHA-256 before it can produce a release artifact.

| Method | Directory | Status |
|---|---|---|
| TEAL-CLIP-FFN | `teal/` | ready |
| OPTIN-CLIP-FFN | `optin/` | ready |
| FLAP-CLIP-FFN | `flap/` | ready |
| MoPE-CLIP-FFN | `mope/` | ready |

Generated thresholds, metrics, and summaries are written under
`outputs/clip_vitl14_text_ffn_baselines/` and are not stored in this code
repository.

Method papers, upstream repositories, and recorded revisions are listed in
the repository-level `THIRD_PARTY.md`.

# Experiment index

This index connects the documented experiment areas with their executable code.

| ID | Experiment | Primary model/data | Release directory | Status |
|---|---|---|---|---|
| Core-V | Table 2 SparMoE-VL | CLIP ViT-L/14, ShareGPT4V, COCO | `experiments/sparmoe_vl_clip_vitl14/vision` | ready |
| Core-T | Table 1 SparMoE-VL | CLIP ViT-L/14, ShareGPT4V, COCO | `experiments/sparmoe_vl_clip_vitl14/text` | ready |
| T1 | Text FFN baselines | CLIP ViT-L/14, ShareGPT4V, COCO/Flickr30k | `experiments/clip_vitl14_text_ffn_baselines` | ready |
| T2 | Vision FFN baselines | CLIP ViT-L/14, ShareGPT4V, COCO/Flickr30k | `experiments/clip_vitl14_vision_ffn_baselines` | ready |
| T3 | Budget Pareto | CLIP ViT-L/14, ShareGPT4V, COCO/Flickr30k | `experiments/sparmoe_vl_clip_vitl14/studies/vision_budget_sweep` | ready |
| T4 | Generalization | COCO/Flickr30k/CIFAR/ImageNet/Food | `experiments/sparmoe_vl_clip_vitl14/studies/cross_dataset_generalization` | ready |
| T5 | LLaVA transfer | LLaVA-v1.5-7B, CLIP-336, POPE/MME-P/GQA/VQAv2 | `experiments/downstream/llava_transfer` | ready |
| T6-CLIP | Architecture transfer: CLIP | ViT-B/16 and ViT-B/32; vision and text | `experiments/architecture_transfer/clip` | ready |
| T6-SigLIP | Architecture transfer: SigLIP | B/16, L/16, SO400M/14; vision and text | `experiments/architecture_transfer/siglip` | ready |
| T6-SigLIP2 | Architecture transfer: SigLIP2 | B/16 and L/16; vision and text | `experiments/architecture_transfer/siglip2` | ready |
| T7 | Capacity intervention | Frozen CLIP ViT-L/14 N=8, ShareGPT4V/COCO | `experiments/sparmoe_vl_clip_vitl14/studies/capacity_intervention` | ready |
| T8 | Component ablation | CLIP ViT-L/14, ShareGPT4V/COCO/Flickr30k | `experiments/sparmoe_vl_clip_vitl14/studies/component_ablation` | ready |
| F3 | Dense FFN capacity requirement | Frozen CLIP ViT-L/14, COCO | `experiments/figures/dense_ffn_capacity_requirement` | ready |
| F4 | Layer-wise expert usage and capacity allocation | CLIP ViT-L/14 Stage 2, COCO | `experiments/figures/layerwise_capacity_allocation` | ready |
| F5 | Input-dependent vision/text token routing | CLIP ViT-L/14 Stage 2, fixed COCO/text inputs | `experiments/figures/input_dependent_token_routing` | ready |
| F6 | Routing granularity and geometry preservation | CLIP ViT-L/14 | `experiments/figures/routing_granularity_geometry_preservation` | ready |
| F7 | Layer-wise local/global representation consistency | CLIP ViT-L/14 Stage 2, COCO | `experiments/figures/layerwise_representation_consistency` | ready |
| F8 | Cross-modal similarity structure preservation | CLIP ViT-L/14 Stage 2, COCO | `experiments/figures/cross_modal_similarity_preservation` | ready |

The status is updated only after the directory has a verified configuration,
executable entry points, an expected output schema, and protocol tests.

## Dense FFN capacity requirement

Figure 3 is reproduced by a frozen Dense CLIP ViT-L/14 analysis, not by a
trained SparMoE-VL checkpoint. Seed 42 partitions all 5,000 COCO val2017
images into 1,000 calibration images and 4,000 non-overlapping evaluation
images. The calibration pass ranks the 4,096 FFN channels separately in all
24 Transformer layers. The evaluation pass records the minimum capacity from
`0.3` through `1.0` that satisfies cosine similarity at least `0.95` and
normalized reconstruction error at most `0.20` for each non-CLS patch token.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/dense_ffn_capacity_requirement/run_study.sh \
  cuda:0
```

The layer shards are validated for model and data identities, split identity,
token-count conservation, and exact coverage of layers 1–24 before merging.
The figure is generated from the merged analysis under ignored `outputs/`;
neither measurements nor reported result values are stored in the source tree.

## Layer-wise capacity allocation

Figure 4 uses the visual main experiment's seed-42 Stage-2 checkpoint at
target ratio `0.7`. The checkpoint must identify the same ordered 500,000-image
ShareGPT4V training pool used by both main training stages. Evaluation then
processes all 5,000 COCO val2017 images in ascending COCO image-ID order and
counts the learned argmax expert for every one of the 256 patch tokens at each
of the 24 Transformer layers.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/layerwise_capacity_allocation/run_study.sh cpu
```

The collector supports protocol-bound resumable counts. It produces one
validated analysis JSON, from which `plot.py` independently renders the expert
usage heatmap and adaptive-capacity panel. Generated measurements, partial
state, and plots remain under ignored
`outputs/figures/layerwise_capacity_allocation`; no reported values are stored
in the code bundle.

## Input-dependent token routing

Figure 5 uses the visual and text main experiment's seed-42 Stage-2
checkpoints. The visual panel routes the same four registered COCO images
at one-based Transformer Layers 6, 12, and 18. The text panel routes the fixed
registered paragraph at one-based Layer 12 while preserving each visible word's BPE
piece assignments. The registered filenames, image bytes, image order, and
text bytes are protected by SHA-256 checks.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/input_dependent_token_routing/run_study.sh cpu
```

The analysis entry point validates that both checkpoints come from their exact
500,000-sample main-training pools before inference. It stores measured routes
only in ignored `outputs/figures/input_dependent_token_routing`; the public
configuration contains protocol and input identities, not experimental
outcomes.

## Routing granularity and geometry preservation

Figure 6 evaluates seed-42 CLIP ViT-L/14 visual models with 4, 6, 8, and 10
nested capacities. Every expert count follows the same protocol: Stage 1
learns SPG, reference capacities, and nested channel subspaces under global
budget P; Stage 2 freezes that initialized structure and optimizes only the
token router within its inherited budget-bounded expert space.
Every path uses data seed 42 and verifies the visual main experiment's exact
ordered 500,000-image ShareGPT4V pool in both stages.

Evaluation follows the documented ordering of all 5,000 COCO val2017
images and all 25,014 captions. For each learned pass it constructs a
capacity-matched anti-assignment and replays those exact expert IDs in a second
pass. Per-layer, per-batch expert histograms and activated FFN MACs must match
exactly; the only intervention is the token-to-capacity correspondence.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/routing_granularity_geometry_preservation/run_training.sh \
  cuda:0
bash experiments/figures/routing_granularity_geometry_preservation/run_study.sh \
  cuda:0
```

Compact weights belong in the corresponding N-specific checkpoint
placeholders. Newly measured retrieval metrics, the generated summary, and
the `routing_geometry_preservation` PDF/PNG remain only under ignored
`outputs/figures/routing_granularity_geometry_preservation`.

## Layer-wise representation consistency

Figure 7 reuses the visual main experiment's seed-42 Stage-2 checkpoint and
the frozen Dense CLIP ViT-L/14 reference. It scans all 5,000 COCO val2017
images in the result-generating lexicographic filename order and observes the
post-block representation at each of the 24 Transformer layers. Learned
argmax routing is used throughout.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/layerwise_representation_consistency/run_study.sh \
  cpu
```

The collector makes a single inference pass and maintains protocol-bound,
resumable float32 memory maps. It stores all patch-token cosine similarities
for the distribution heatmap and both Dense/SparMoE CLS states for exact
centered linear CKA; it also accumulates CLS and all-token cosine means.
Generated memory maps, statistics, and the semantic
`layerwise_representation_consistency` PDF/PNG stay only under ignored
`outputs/figures/layerwise_representation_consistency`.

## Cross-modal similarity structure preservation

Figure 8 uses the validated final-layer Dense and SparMoE image CLS states
from Figure 7, then applies the frozen CLIP visual `ln_post` and projection.
Both image branches are compared against exactly the same frozen Dense CLIP
text tower; no sparse text checkpoint is part of this experiment. The full
streaming calculation covers the complete 5,000-by-25,014 COCO val2017
image-text similarity matrix without storing that large matrix.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/figures/cross_modal_similarity_preservation/run_study.sh \
  cuda:0
```

The visualization uses a uniform score-independent sample of 256 images with
seed 42 and each image's first registered caption. Average-linkage clustering
of the Dense joint image-text embeddings determines one shared display order;
the Dense and SparMoE matrices are then aggregated into the same 64-by-64 grid
with one shared color range. Feature caches, statistics, sample manifests, and
the semantic PDF/PNG remain under ignored
`outputs/figures/cross_modal_similarity_preservation`.

## LLaVA transfer

The downstream study uses a separately converted CLIP ViT-L/14-336 vision
tower because LLaVA-v1.5 selects 576 patch features from visual layer -2. Its
Stage 1 aligns all 24 patch-token hidden states and the final pooled feature;
Stage 2 freezes the SPG structure and learns only nested token-conditioned
routing within the inherited global budget. Both stages use the exact
500,000-image ordered ShareGPT4V pool of the visual main experiment (data seed
42 and SHA-256
`f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0`)
for each of training seeds 42, 123, and 2026.

Run `experiments/downstream/llava_transfer/validate_data.py` before evaluation.
Then use `run_dense_reference.sh` once and `run_sparse_seed.sh` for each sparse
checkpoint, or use `run_full_study.sh` for the complete workflow. The summary
command rejects incomplete benchmarks, mixed checkpoints, changed benchmark
identities, and sparse runs whose training-data identity differs from the
registered protocol.

```bash
conda activate sparmoe-vl
python3 experiments/downstream/llava_transfer/validate_data.py

# Train the three two-stage CLIP-336 conversions.
bash experiments/downstream/llava_transfer/run_training.sh cuda:0

# Evaluate newly trained weights, or omit CHECKPOINT_ROOT to use placeholders.
CHECKPOINT_ROOT=outputs/downstream/llava_transfer/training \
  bash experiments/downstream/llava_transfer/run_full_study.sh cuda:0
```

The Dense reference is evaluated once. Each sparse seed is evaluated on the
same benchmark identities, and VQAv2 is split only for execution: its four
fixed contiguous slices are merged and checked before aggregation.

## SigLIP2 architecture transfer

The SigLIP2 rows of Table 6 are versioned as `siglip2/vit_b16` and
`siglip2/vit_l16`. Each version contains dedicated Stage-1, Stage-2, retrieval,
three-seed aggregation, and orchestration entry points for both the vision and
text towers. The two model versions intentionally share one SigLIP-family
controller implementation because their OpenCLIP tower contracts are the same;
their registered model identities, geometry, local weights, tokenizer, batch
sizes, checkpoints, and output paths remain separate.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/architecture_transfer/siglip2/vit_b16/run_three_seeds.sh \
  vision cuda:0
bash experiments/architecture_transfer/siglip2/vit_l16/run_three_seeds.sh \
  text cuda:0
```

The timm visual trunk has no CLS token, so every patch takes the routed FFN
path. The text tower keeps only its final pooled position on the original dense
FFN path. Both stages use and verify the same modality-specific ordered
500,000-sample ShareGPT4V pool as the main experiment. Generated checkpoints,
retrieval metrics, and summaries are written only beneath ignored `outputs/`;
`checkpoints/architecture_transfer/siglip2` contains release placeholders.

## SigLIP architecture transfer

The SigLIP part of the architecture-transfer study is separated into the exact
registered versions `siglip/vit_b16`, `siglip/vit_l16`, and `siglip/so400m14`.
Each version directory fixes its own model identity, geometry, local assets,
result-generating batch sizes, and complete workflow while reusing one verified
implementation under `src/`. Every model is run on both the vision and text
tower with seeds 42, 123, and 2026. Every run uses 500,000 ShareGPT4V samples
selected with data seed 42; Stage 2 rejects a Stage-1 checkpoint with a
different ordered fingerprint.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/architecture_transfer/siglip/vit_b16/run_three_seeds.sh \
  vision cuda:0
bash experiments/architecture_transfer/siglip/vit_l16/run_three_seeds.sh \
  text cuda:0
bash experiments/architecture_transfer/siglip/so400m14/run_three_seeds.sh \
  vision cuda:0
```

The visual implementation routes every patch because SigLIP has no CLS token.
The text implementation keeps the last pooled position dense and routes all
preceding positions. COCO val2017 and Flickr30k test R@1 are evaluated in full;
generated checkpoints, metrics, and summaries remain under ignored `outputs/`.

## CLIP architecture transfer

The CLIP part of Table 6 is separated by the exact backbone version:
`clip/vit_b16` and `clip/vit_b32`. Each directory fixes its own OpenCLIP model
name, patch size, architecture dimensions, and result-generating batch sizes.
CLIP ViT-L/14 is intentionally absent because it is the main experiment, not
an architecture-transfer point.

Both towers retain CLIP-specific semantics: visual CLS and text position zero
take the original dense FFN path, while all remaining positions are routed.
For every seed, both stages rebuild and verify the same 500,000-sample ordered
ShareGPT4V pool used by the corresponding main-experiment modality.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/architecture_transfer/clip/vit_b16/run_three_seeds.sh \
  vision cuda:0
bash experiments/architecture_transfer/clip/vit_b32/run_three_seeds.sh \
  text cuda:0
```

The evaluation entry points accept only the documented release checkpoint
schema. New checkpoints and all generated metrics remain under ignored
`outputs/`; release-weight locations under `checkpoints/architecture_transfer/clip`
are placeholders only.

## Capacity intervention

Table 7 uses a dedicated N=8 visual model. Its Stage 1 learns the N=8 SPG,
reference capacities, and nested subspaces under global budget P; its Stage 2
freezes them and optimizes only the N=8 token router within that
budget-bounded structure. Both stages use the visual
main experiment's exact ordered 500,000-image ShareGPT4V pool (data seed 42;
SHA-256
`f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0`).
Training seeds are 42, 123, and 2026.

For every seed, evaluation computes the Dense reference once on all 5,000
COCO val2017 images and then applies Self, Uniform, Shuffled, Layer-Uniform,
and Layer-Shuffled assignments. Uniform and shuffled assignments preserve the
measured comparison budget used by the original result-generating code.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/sparmoe_vl_clip_vitl14/studies/capacity_intervention/run_three_seeds.sh \
  cuda:0
```

The fixed protocol is visible without starting training through
`train_stage1.py --seed 42 --check-only` or
`train_stage2.py --seed 42 --check-only`. The corresponding `evaluate.py --check-only`
also validates the checkpoint identity and the complete COCO annotation set.
Weights may be placed under
`checkpoints/sparmoe_vl_clip_vitl14/studies/capacity_intervention`; generated
checkpoints, per-seed metrics, and summaries remain under ignored `outputs/`.

## Component ablation

Table 8 has one explicit implementation for each documented row. `w/o SPG`,
`w/o Layer-Adaptive Budget`, and `w/o Geometry Preservation` are separately
trained from frozen Dense CLIP. All three reconstruct and hard-check the visual
main experiment's ordered 500,000-image ShareGPT4V pool (data seed 42;
SHA-256
`f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0`). The
last removal is called
`no_feature_distillation` in the raw run because it replaces Dense feature
distillation with frozen CLIP image-text contrastive supervision.

`w/o Token Router` is not a separately trained static router. The documented row
reuses each visual main Stage-2 checkpoint and replaces its learned decisions
with uniform random expert choices per patch token at evaluation. COCO uses
routing seed 42 and Flickr30k uses 43. The full row reuses the same main
checkpoints with learned routing.

```bash
conda activate sparmoe-vl
pip install -e .
bash experiments/sparmoe_vl_clip_vitl14/studies/component_ablation/run_study.sh \
  cuda:0
```

The geometry-preservation row uses its registered historical
seeds 42, 123, and 3407; the other rows use 42, 123, and 2026. This exception
is declared in `config.yaml` and validated by the summarizer. All generated
weights and metrics remain beneath ignored `outputs/`.

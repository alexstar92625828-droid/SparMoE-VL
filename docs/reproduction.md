# Reproducing the documented experiments

This guide gives the intended execution order for the complete SparMoE-VL
artifact. Commands are run from the repository root after installation.
Generated files always remain under the Git-ignored `outputs/` directory.

## 1. Prepare the environment and assets

Install the project and arrange the external assets described in
[data_layout.md](data_layout.md). A repository and its assets can live on any
disks by setting:

```bash
export SPARMOE_VL_ROOT=/path/to/SparMoE-VL
export SPARMOE_VL_WORKSPACE=/path/to/external-assets
```

Before a costly run, inspect the relevant `config.yaml` and use an entry
point's `--help` or `--check-only` mode where available. Identity checks are
deliberately strict: changing a registered dataset, seed, sample order,
checkpoint stage, or model version fails instead of producing a silently
incomparable result.

## 2. Train the main CLIP ViT-L/14 models

The main visual and text experiments each use 500,000 ShareGPT4V samples,
data seed 42, and training seeds 42, 123, and 2026. Stage 2 reconstructs and
verifies the exact ordered Stage-1 pool. Stage 1 trains only SPG channel
orderings, layer-wise reference capacities, and nested subspaces under global
budget `p`. Stage 2 initializes from and freezes that structure, then
optimizes only the token routers. The learned nested experts keep the Stage-1
budget as a structural upper bound.

```bash
bash experiments/sparmoe_vl_clip_vitl14/run_three_seeds.sh vision cuda:0
bash experiments/sparmoe_vl_clip_vitl14/run_three_seeds.sh text cuda:0
```

For manual or multi-device scheduling, launch the `train_stage1.py`,
`train_stage2.py`, and `evaluate.py` files inside `vision/` or `text/`
individually. Stage 2 must receive the matching Stage-1 checkpoint.

## 3. Run the FFN baselines

Each comparison method is isolated by modality and method. All eight
directories consume the same modality-specific ordered 500,000-sample pool as
the corresponding main experiment.

```bash
bash experiments/clip_vitl14_text_ffn_baselines/teal/run_three_seeds.sh
bash experiments/clip_vitl14_text_ffn_baselines/optin/run_three_seeds.sh
bash experiments/clip_vitl14_text_ffn_baselines/flap/run_three_seeds.sh
bash experiments/clip_vitl14_text_ffn_baselines/mope/run_three_seeds.sh

bash experiments/clip_vitl14_vision_ffn_baselines/teal/run_three_seeds.sh
bash experiments/clip_vitl14_vision_ffn_baselines/optin/run_three_seeds.sh
bash experiments/clip_vitl14_vision_ffn_baselines/flap/run_three_seeds.sh
bash experiments/clip_vitl14_vision_ffn_baselines/mope/run_three_seeds.sh
```

These methods have different preparation phases by design. Their launchers
run calibration or structure selection, optional recovery, evaluation, and
aggregation in the protocol-defined order.

## 4. Reproduce the remaining tables

The budget sweep reuses the main visual `p=0.7` runs and trains the remaining
registered target ratios:

```bash
bash experiments/sparmoe_vl_clip_vitl14/studies/vision_budget_sweep/run_sweep.sh \
  cuda:0
```

Cross-dataset evaluation reuses the main visual and text checkpoints:

```bash
bash experiments/sparmoe_vl_clip_vitl14/studies/cross_dataset_generalization/run_three_seeds.sh \
  cuda:0
```

The downstream, architecture-transfer, intervention, and ablation workflows
are launched independently:

```bash
bash experiments/downstream/llava_transfer/run_training.sh cuda:0
CHECKPOINT_ROOT=outputs/downstream/llava_transfer/training \
  bash experiments/downstream/llava_transfer/run_full_study.sh cuda:0

bash experiments/architecture_transfer/clip/vit_b16/run_three_seeds.sh vision cuda:0
bash experiments/architecture_transfer/clip/vit_b32/run_three_seeds.sh text cuda:0
bash experiments/architecture_transfer/siglip/vit_b16/run_three_seeds.sh vision cuda:0
bash experiments/architecture_transfer/siglip/vit_l16/run_three_seeds.sh text cuda:0
bash experiments/architecture_transfer/siglip/so400m14/run_three_seeds.sh vision cuda:0
bash experiments/architecture_transfer/siglip2/vit_b16/run_three_seeds.sh vision cuda:0
bash experiments/architecture_transfer/siglip2/vit_l16/run_three_seeds.sh text cuda:0

bash experiments/sparmoe_vl_clip_vitl14/studies/capacity_intervention/run_three_seeds.sh \
  cuda:0
bash experiments/sparmoe_vl_clip_vitl14/studies/component_ablation/run_study.sh \
  cuda:0
```

Run every architecture version for both `vision` and `text`; the abbreviated
commands above show the interface without duplicating every combination.

## 5. Reproduce the mechanism figures

The figure directories are named by the scientific question, not by their
numbered labels. Run them in the following dependency order:

```bash
bash experiments/figures/dense_ffn_capacity_requirement/run_study.sh cuda:0
bash experiments/figures/layerwise_capacity_allocation/run_study.sh cuda:0
bash experiments/figures/input_dependent_token_routing/run_study.sh cuda:0

bash experiments/figures/routing_granularity_geometry_preservation/run_training.sh cuda:0
bash experiments/figures/routing_granularity_geometry_preservation/run_study.sh cuda:0

bash experiments/figures/layerwise_representation_consistency/run_study.sh cuda:0
bash experiments/figures/cross_modal_similarity_preservation/run_study.sh cuda:0
```

The cross-modal similarity analysis consumes the validated final-layer
features from the layer-wise representation analysis. Other figure workflows
are independent apart from their documented pretrained or SparMoE-VL
checkpoint inputs.

## 6. Validate the code archive

The public test suite does not download data or models:

```bash
python -m pip install -e ".[analysis,dev]"
ruff format --check .
ruff check .
pytest
```

The experiment-to-directory mapping and detailed study contracts are listed in
[experiment_index.md](experiment_index.md).

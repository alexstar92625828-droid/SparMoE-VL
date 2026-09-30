# SparMoE-VL on CLIP ViT-L/14

This directory contains the main SparMoE-VL experiments, the visual budget
study, and the CLIP ViT-L/14 studies. The main method uses the two-stage
protocol documented in this repository.

Create the pinned reference environment from the repository root:

```bash
conda env create -f environment.yml
conda activate sparmoe-vl
python -m pip install -e .
```

## Two-stage contract

1. Stage 1 learns the SPG channel-priority ordering, one reference capacity
   per Transformer layer, and the resulting nested FFN subspaces. The mean
   layer reference capacity is constrained by the global budget `p`.
2. Stage 2 loads the complete Stage-1 structure—including the SPG, layer
   reference capacities, channel orderings, and nested masks—freezes it, and
   optimizes only the token routers. The fixed nested experts inherit the
   Stage-1 global budget as a structural upper bound.
3. Stage 2 reconstructs the training subset with the same `data_seed` and
   checks its ordered SHA-256 against the Stage-1 checkpoint.  A mismatch is a
   hard error.  Thus a run cannot silently use different data in the two
   stages.

The reference settings are vision `p=0.7`, text `p=0.6`, four capacity factors
`[0.7, 0.8, 0.9, 1.0]`, 500,000 maximum ShareGPT4V samples, 5,000 steps per
stage, data seed 42, and training seeds 42/123/2026.

## Run one seed

Run from the repository root after installing the package.

```bash
python3 experiments/sparmoe_vl_clip_vitl14/vision/train_stage1.py \
  --seed 42 --output-dir outputs/main/vision/seed_42/stage1

python3 experiments/sparmoe_vl_clip_vitl14/vision/train_stage2.py \
  --seed 42 \
  --stage1-checkpoint outputs/main/vision/seed_42/stage1/best.pt \
  --output-dir outputs/main/vision/seed_42/stage2

python3 experiments/sparmoe_vl_clip_vitl14/vision/evaluate.py \
  --checkpoint outputs/main/vision/seed_42/stage2/best.pt
```

Replace `vision` with `text` for the text-side experiment. The
`run_three_seeds.sh` script runs both stages for all registered seeds. Checkpoint
metadata and the fixed protocol configuration provide the reproducibility
record used by the public entry points.

For one visual seed per GPU, the resource-aware controller waits until three
GPUs are genuinely available, launches all registered seeds together, retries only
GPU-contention failures, resumes completed stages, evaluates every seed, and
then writes an aggregate summary under the ignored output directory:

```bash
mkdir -p outputs/main/vision/controller
nohup setsid env PYTHON_BIN="$(command -v python)" \
  bash experiments/sparmoe_vl_clip_vitl14/vision/run_three_gpu_pipeline.sh \
  > outputs/main/vision/controller/controller.log 2>&1 < /dev/null &
```

## Visual budget study

The Table 3 / Figure 2 entry points live under `studies/vision_budget_sweep`.
They train `p=0.4/0.5/0.6/0.8` with the visual main protocol and reuse the
three visual-main checkpoints for `p=0.7`, according to the registered workflow.

```bash
bash experiments/sparmoe_vl_clip_vitl14/studies/vision_budget_sweep/run_sweep.sh cuda:0
```

All checkpoints, retrieval metrics, tables, and plots are generated below
`outputs/studies/vision_budget_sweep/`, which is excluded from the code bundle.

The complete Table 4 evaluation is under
`studies/cross_dataset_generalization`. It reuses the visual `p=0.7` and text
`p=0.6` main checkpoints on COCO, Flickr30k, CIFAR-100, ImageNet-1K, and
Food-101; it does not train an additional model.

## Capacity intervention study

Table 7 is under `studies/capacity_intervention`. It applies the same two-stage
SPG-then-router protocol to an N=8 visual model. It uses the visual main
experiment's exact ordered 500,000-image ShareGPT4V pool and evaluates all
five capacity interventions on all 5,000 COCO val2017 images.

```bash
bash experiments/sparmoe_vl_clip_vitl14/studies/capacity_intervention/run_three_seeds.sh \
  cuda:0
```

The script trains seeds 42, 123, and 2026, evaluates each selected checkpoint,
and produces the token- and layer-level aggregate tables under the ignored
`outputs/sparmoe_vl_clip_vitl14/studies/capacity_intervention` directory.

## Component ablation study

Table 8 is under `studies/component_ablation`. Its three trainable removals
start directly from the same frozen Dense CLIP and use the visual main
experiment's exact ordered 500,000-image pool. The Token Router removal is an
inference-only random-routing intervention on the main Stage-2 checkpoints;
the complete row also reuses those checkpoints.

```bash
bash experiments/sparmoe_vl_clip_vitl14/studies/component_ablation/run_study.sh \
  cuda:0
```

The fixed settings and method-to-replacement mapping are in `config.yaml`.
Generated weights, per-seed evaluations, and the aggregate table are written
only under the ignored `outputs/sparmoe_vl_clip_vitl14/studies/component_ablation`
directory.

## Dense FFN capacity requirement

Figure 3 is archived separately under
`experiments/figures/dense_ffn_capacity_requirement`. This is a
Dense-only frozen-model analysis: it uses no SparMoE-VL checkpoint and trains
no parameters. The complete COCO val2017 set is deterministically divided
into 1,000 calibration images and 4,000 disjoint evaluation images. The
calibration subset defines one shared FFN-channel ranking per layer; the
evaluation subset measures the minimum registered width meeting both
reconstruction criteria for every patch token.

```bash
bash experiments/figures/dense_ffn_capacity_requirement/run_study.sh \
  cuda:0
```

The computation is split into layers 1–12 and 13–24 to bound GPU memory. The
merger rejects missing, duplicated, or protocol-incompatible layers before
the plotting entry point can run. Measurements and the generated PDF/PNG are
written only beneath the ignored
`outputs/figures/dense_ffn_capacity_requirement` path.

The visual main checkpoint is also consumed by the separately archived
layer-wise mechanism analysis under
`experiments/figures/layerwise_capacity_allocation`. That directory owns its
COCO routing collector, capacity derivation, plotting entry point, and ignored
output path; no duplicate checkpoint is created for the analysis.

The same seed-42 visual Stage-2 checkpoint is reused by the independent
representation analysis under
`experiments/figures/layerwise_representation_consistency`. It compares
post-block Dense and sparse representations across all 24 layers on the full
COCO val2017 set, while all memory maps, measurements, and plots remain under
the ignored `outputs/figures/layerwise_representation_consistency` directory.

Those validated final-layer CLS maps are consumed by
`experiments/figures/cross_modal_similarity_preservation`. The latter compares
Dense and SparMoE visual features against one shared frozen Dense text tower
over the complete COCO val2017 image-caption matrix; it does not introduce a
new training run or a sparse text checkpoint.

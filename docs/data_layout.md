# Data and model layout

Datasets and upstream model weights are not committed to this repository.
They remain subject to their publishers' licenses and access conditions.

## Recommended workspace

The default resolver expects a source checkout next to an external-assets
tree with the following layout:

```text
workspace/
├── SparMoE-VL/
├── models/
│   ├── ViT-L-14.pt
│   ├── clip-vit-large-patch14-336/
│   ├── llava-v1.5-7b-hf-safetensors/
│   └── llava-v1.5-7b/
├── ShareGPT4V/
│   ├── annotations/sharegpt4v_1246k.json
│   └── images/
└── data/eval/
    ├── coco/
    ├── flickr30k/
    ├── cifar100/
    ├── imagenet/
    ├── food101/
    ├── pope/
    ├── mme/
    ├── gqa/
    └── vqav2/
```

The checkout and assets may instead live anywhere. Configure both roots once
for a local machine, compute node, or container:

```bash
export SPARMOE_VL_ROOT=/path/to/SparMoE-VL
export SPARMOE_VL_WORKSPACE=/path/to/workspace
```

`SPARMOE_VL_ROOT` controls repository-relative checkpoints and generated
outputs. `SPARMOE_VL_WORKSPACE` controls the default locations of external
models and datasets. Individual command-line options take precedence whenever
an experiment exposes a path directly.

## Training data identity

The main vision and text experiments construct separate ordered pools of
500,000 ShareGPT4V examples with data seed 42. Every comparison and transfer
study that claims the main training data validates the corresponding ordered
pool SHA-256. Stage 2 also checks the Stage-1 sample count, data seed, and
ordered identity before optimization begins.

Do not rename or substitute files after generating a run's dataset identity.
If an upstream dataset is stored through symlinks or mounted storage, pass the
resolved root consistently to all stages.

## LLaVA assets

For the LLaVA transfer study, `llava-v1.5-7b-hf-safetensors` contains the
model configuration and sharded model weights, while `llava-v1.5-7b` supplies
the matching tokenizer. `clip-vit-large-patch14-336` contains the original
336-pixel CLIP vision configuration and weights.

Run the public validation entry point before downstream inference:

```bash
python experiments/downstream/llava_transfer/validate_data.py
```

Benchmark annotation files are checked against the SHA-256 identities in
`experiments/downstream/llava_transfer/config.yaml`.

## Generated files

New checkpoints, feature caches, predictions, metrics, temporary memory maps,
and plots belong under `outputs/`. The directory and common weight formats are
ignored by Git. Optional released weights follow the separate
[checkpoint layout](../checkpoints/README.md).

# Checkpoint layout

This directory records where optional pretrained SparMoE-VL weights should be
placed. Binary checkpoints are intentionally excluded from Git; distribute
them through a model host or GitHub release assets and preserve their published
SHA-256 checksums.

```text
checkpoints/
├── sparmoe_vl_clip_vitl14/
│   ├── vision/seed_{42,123,2026}/stage2/best.pt
│   ├── text/seed_{42,123,2026}/stage2/best.pt
│   └── studies/
│       ├── capacity_intervention/seed_{42,123,2026}/
│       │   ├── stage1/best.pt
│       │   └── stage2/best.pt
│       ├── component_ablation/
│       │   ├── without_spg/seed_{42,123,2026}/{stage1,stage2}/best.pt
│       │   ├── without_layer_adaptive_budget/seed_{42,123,2026}/{stage1,stage2}/best.pt
│       │   └── without_geometry_preservation/seed_{42,123,3407}/{stage1,stage2}/best.pt
│       └── routing_granularity_geometry_preservation/
│           ├── n4/{stage1,stage2}/best.pt
│           ├── n6/{stage1,stage2}/best.pt
│           ├── n8/{stage1,stage2}/best.pt
│           └── n10/{stage1,stage2}/best.pt
├── sparmoe_vl_clip336_llava/
│   └── seed_{42,123,2026}/stage2/best.pt
└── architecture_transfer/
    ├── clip/{vit_b16,vit_b32}/{vision,text}/seed_{42,123,2026}.pt
    ├── siglip/{vit_b16,vit_l16,so400m14}/{vision,text}/seed_{42,123,2026}.pt
    └── siglip2/{vit_b16,vit_l16}/{vision,text}/seed_{42,123,2026}.pt
```

Both structural Stage-1 and router-only Stage-2 locations are placeholders.
Stage-2 checkpoints retain the frozen Stage-1 structure together with the
learned token-router weights. Binary weights remain external to Git and may be
published through release assets or a model host.

Release checkpoints use the compact schemas written by the public training
code: learned controllers, sparse projections, routers, budgets, and protocol
metadata are stored without duplicating frozen upstream backbone tensors. The
evaluators validate modality, architecture, stage, seed, capacity factors,
training-pool identity, and expected tensor structure before loading them.

Some documented studies deliberately reuse another checkpoint:

- the budget sweep's `p=0.7` row uses the visual main Stage-2 weights;
- two component-ablation rows use the visual main Stage-2 weights;
- mechanism figures reuse the documented seed-42 main weight or a preceding
  analysis artifact.

No measured metrics, predictions, or plots belong in this directory. They are
always generated under the Git-ignored `outputs/` tree.

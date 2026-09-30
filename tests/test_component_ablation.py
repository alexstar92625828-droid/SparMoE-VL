import pytest
import torch
from torch import nn

from sparmoe_vl.architecture_transfer.clip.model import NestedCLIPFFN
from sparmoe_vl.common.two_stage import STAGE1_PROTOCOL, STAGE2_PROTOCOL
from sparmoe_vl.studies.component_ablation.checkpoints import checkpoint_metadata
from sparmoe_vl.studies.component_ablation.data import extract_contrastive_caption
from sparmoe_vl.studies.component_ablation.evaluation import ffn_macs
from sparmoe_vl.studies.component_ablation.model import (
    enable_independent_expert_masks,
    freeze_layer_adaptive_budget,
)
from sparmoe_vl.studies.component_ablation.protocol import (
    CAPACITY_FACTORS,
    DATASET_SHA256,
    EVALUATION_COUNTS,
    EVALUATION_SEED,
    EVALUATION_SHA256,
    METHODS,
    METHOD_LABELS,
    MODEL_KEY,
    MODEL_NAME,
    PAPER_SEEDS,
    REPLACEMENTS,
    training_manifest,
)
from sparmoe_vl.studies.component_ablation.summary import (
    markdown_table,
    summarize_results,
)
from sparmoe_vl.studies.component_ablation.training import parse_args


def release_checkpoint(method: str, seed: int = 42, stage: int = 2) -> dict:
    layers = {}
    if method == "without_spg":
        layers.update(
            {f"{layer}.independent_mask_logits": torch.empty(4, 8) for layer in range(24)}
        )
    if method == "without_layer_adaptive_budget":
        fixed = torch.logit(torch.tensor(0.7))
        layers.update({f"{layer}.full_ratio_logit": fixed.clone() for layer in range(24)})
    if stage == 2:
        layers["0.router.weight"] = torch.empty(4, 8)
    checkpoint = {
        "format_version": 3,
        "method": "sparmoe_vl_component_ablation",
        "study": "component_ablation",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "step": 1_500,
        "training_seed": seed,
        "training_protocol": STAGE1_PROTOCOL if stage == 1 else STAGE2_PROTOCOL,
        "stage": stage,
        "modality": "vision",
        "variant": method,
        "replacement": REPLACEMENTS[method],
        "target_ratio": 0.7,
        "capacity_factors": list(CAPACITY_FACTORS),
        "dataset": {
            "data_seed": 42,
            "samples": 500_000,
            "ordered_sha256": DATASET_SHA256,
        },
        "controller": {"hypernetwork": {}, "layers": layers},
    }
    if stage == 2:
        checkpoint["stage1"] = {
            "protocol": STAGE1_PROTOCOL,
            "dataset_sha256": DATASET_SHA256,
        }
    return checkpoint


@pytest.mark.parametrize(
    "method",
    (
        "without_spg",
        "without_layer_adaptive_budget",
        "without_geometry_preservation",
    ),
)
def test_release_checkpoints_require_the_exact_500k_protocol(method: str) -> None:
    metadata = checkpoint_metadata(release_checkpoint(method), method)
    assert metadata["format"] == "release_v3"
    assert metadata["pool_size"] == 500_000
    changed = release_checkpoint(method)
    changed["dataset"]["samples"] = 50_000
    with pytest.raises(ValueError, match="dataset.samples"):
        checkpoint_metadata(changed, method)


def test_paper_seed_mapping_preserves_the_result_generating_exception() -> None:
    assert PAPER_SEEDS["without_geometry_preservation"] == (42, 123, 3407)
    assert all(
        seeds == (42, 123, 2026)
        for method, seeds in PAPER_SEEDS.items()
        if method != "without_geometry_preservation"
    )


def test_component_ablation_freezes_each_stage1_replacement_in_stage2() -> None:
    stage1 = training_manifest("without_spg", 42, 1)
    stage2 = training_manifest("without_spg", 42, 2)
    assert "global_budget_p" in stage1["objective"]
    assert "global_budget_p" not in stage2["objective"]
    assert "token_router" not in stage1["trainable"]
    assert stage2["trainable"] == ["token_router"]
    assert "spg_or_registered_replacement" in stage2["frozen"]


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(8, 16)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(inputs)))


class ToyAblationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(index): NestedCLIPFFN(ToyMLP(), 8, 16, CAPACITY_FACTORS, 0.7)
                for index in range(2)
            }
        )
        self.hypernetwork = nn.Linear(3, 4)


def test_without_spg_uses_reproducible_independent_expert_masks() -> None:
    first = ToyAblationModel()
    second = ToyAblationModel()
    second.load_state_dict(first.state_dict())
    first_parameters = enable_independent_expert_masks(first, seed=42)
    second_parameters = enable_independent_expert_masks(second, seed=42)
    assert len(first_parameters) == len(first.layers)
    assert all(not parameter.requires_grad for parameter in first.hypernetwork.parameters())
    for left, right, layer in zip(
        first_parameters,
        second_parameters,
        first.layers.values(),
    ):
        assert left.shape == (4, 16)
        assert torch.equal(left, right)
        layer.eval()
        masks, ratios = layer.budget_masks(torch.empty(4, 128), 0.4)
        assert masks.sum(-1).tolist() == [8.0, 9.0, 10.0, 11.0]
        assert ratios.tolist() == pytest.approx([0.49, 0.56, 0.63, 0.70])


def test_without_layer_adaptive_budget_fixes_every_layer_at_point_seven() -> None:
    model = ToyAblationModel()
    freeze_layer_adaptive_budget(model)
    for layer in model.layers.values():
        assert float(layer.full_ratio()) == pytest.approx(0.7)
        assert layer.full_ratio_logit.requires_grad is False


def test_caption_selection_matches_the_historical_contrastive_ablation() -> None:
    assert extract_contrastive_caption({"caption": "direct"}) == "direct"
    assert (
        extract_contrastive_caption(
            {"conversations": [{"from": "gpt", "value": "<image> a cat"}]}
        )
        == "a cat"
    )


def canonical_result(method: str, seed: int, value: float) -> dict:
    return {
        "study": "component_ablation",
        "paper_scope": "Table 8",
        "method": method,
        "label": METHOD_LABELS[method],
        "replacement": REPLACEMENTS[method],
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": seed,
        "data_seed": 42,
        "dataset_sha256": DATASET_SHA256,
        "checkpoint_step": seed,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_sha256": EVALUATION_SHA256,
        "counts": EVALUATION_COUNTS,
        "routing": (
            "uniform_random_per_patch_token"
            if method == "without_token_router"
            else "learned_argmax"
        ),
        "dense_reference": {
            "ffn_macs_v_g": 10.0,
            "coco_i2t_r1": 10.0,
            "coco_t2i_r1": 10.0,
            "flickr_i2t_r1": 10.0,
            "flickr_t2i_r1": 10.0,
        },
        "metrics": {
            "ffn_macs_v_g": value,
            "coco_i2t_r1": value,
            "coco_t2i_r1": value,
            "flickr_i2t_r1": value,
            "flickr_t2i_r1": value,
            "retention_percent": value,
        },
    }


def test_summary_uses_per_method_paper_seeds_and_sample_sd() -> None:
    results = {
        method: [
            canonical_result(method, seed, value)
            for seed, value in zip(PAPER_SEEDS[method], (1.0, 2.0, 3.0))
        ]
        for method in METHODS
    }
    summary = summarize_results(results)
    assert summary["method_seeds"]["without_geometry_preservation"] == [42, 123, 3407]
    assert summary["dense_reference"]["coco_i2t_r1"] == 10.0
    for row in summary["methods"].values():
        assert row["retention_percent"]["mean"] == pytest.approx(2.0)
        assert row["retention_percent"]["sample_sd"] == pytest.approx(1.0)
    assert "| Dense |" in markdown_table(summary)
    assert "w/o Token Router" in markdown_table(summary)


def test_visual_ffn_macs_include_one_dense_cls_token_per_layer() -> None:
    expected = (24 + 256 * 24) * 2 * 1024 * 4096 / 1e9
    assert ffn_macs([1.0] * 24) == pytest.approx(expected)
    with pytest.raises(ValueError, match="24 layer ratios"):
        ffn_macs([1.0] * 23)


def test_training_defaults_write_generated_weights_only_under_outputs() -> None:
    args = parse_args(["--variant", "without_spg", "--seed", "42"])
    assert "outputs/sparmoe_vl_clip_vitl14/studies/component_ablation" in str(args.output_dir)
    assert "checkpoints" not in str(args.output_dir)

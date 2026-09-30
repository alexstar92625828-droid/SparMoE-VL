from types import SimpleNamespace

import torch
from torch import nn

from sparmoe_vl.common.budgets import LayerAdaptiveBudget
from sparmoe_vl.common.losses import (
    LayerRoutingLossInput,
    Stage1SubspaceObjective,
    Stage2RouterObjective,
)
from sparmoe_vl.common.routing import TokenCapacityRouter
from sparmoe_vl.common.sparse_patterns import SparsePatternGenerator
from sparmoe_vl.common.two_stage import (
    STAGE1_FROZEN,
    STAGE1_LEARNS,
    STAGE1_PROTOCOL,
    STAGE2_FROZEN,
    STAGE2_LEARNS,
    STAGE2_PROTOCOL,
    TWO_STAGE_PROTOCOL,
)
from sparmoe_vl.text.encoder import SparMoETextEncoder
from sparmoe_vl.vision.encoder import SparMoEVisionEncoder


class FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 8)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(inputs)))


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMLP()


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([FakeBlock(), FakeBlock()])
        self.batch_first = True

    @staticmethod
    def get_cast_dtype() -> torch.dtype:
        return torch.float32


class FakeVisionCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = SimpleNamespace(transformer=FakeTransformer())
        self.add_module("visual_transformer", self.visual.transformer)
        self.visual._embeds = lambda images: torch.zeros(images.shape[0], 2, 4)
        self.visual._pool = lambda states: (states[:, 0], None)
        self.visual.proj = None

    @staticmethod
    def encode_image(images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return torch.ones(images.shape[0], 4)


class FakeTextCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = FakeTransformer()
        self.token_embedding = nn.Embedding(16, 4)
        self.positional_embedding = nn.Parameter(torch.zeros(5, 4))
        self.attn_mask = None
        self.ln_final = nn.Identity()
        self.text_projection = None

    def encode_text(self, tokens: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.token_embedding(tokens).mean(dim=1)


def trainable_names(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def assert_stage_parameter_contract(model: nn.Module, stage: int) -> None:
    names = trainable_names(model)
    assert names
    assert not any(name.startswith("clip_model.") for name in names)
    if stage == 1:
        assert any(name.startswith("budget.") for name in names)
        assert any(name.startswith("sparse_pattern_generator.") for name in names)
        assert not any(name.startswith("routers.") for name in names)
    else:
        assert any(name.startswith("routers.") for name in names)
        assert not any(name.startswith("budget.") for name in names)
        assert not any(name.startswith("sparse_pattern_generator.") for name in names)
        assert all(name.startswith("routers.") for name in names)


def test_canonical_two_stage_contract() -> None:
    assert TWO_STAGE_PROTOCOL == "spg_global_budget_then_frozen_spg_token_router"
    assert STAGE1_PROTOCOL == "spg_global_budget_stage1"
    assert STAGE2_PROTOCOL == "frozen_spg_token_router_stage2"
    assert STAGE1_LEARNS == (
        "sparse_pattern_generator",
        "channel_importance",
        "layer_reference_capacities",
        "nested_expert_subspaces",
    )
    assert STAGE1_FROZEN == ("pretrained_backbone", "token_router")
    assert STAGE2_LEARNS == ("token_router",)
    assert STAGE2_FROZEN == ("pretrained_backbone", *STAGE1_LEARNS)


def test_vision_stage_parameter_contract() -> None:
    assert_stage_parameter_contract(SparMoEVisionEncoder(FakeVisionCLIP(), training_stage=1), 1)
    assert_stage_parameter_contract(SparMoEVisionEncoder(FakeVisionCLIP(), training_stage=2), 2)


def test_text_stage_parameter_contract() -> None:
    assert_stage_parameter_contract(SparMoETextEncoder(FakeTextCLIP(), training_stage=1), 1)
    assert_stage_parameter_contract(SparMoETextEncoder(FakeTextCLIP(), training_stage=2), 2)


def test_stage2_loads_structure_without_loading_stage1_router() -> None:
    stage1 = SparMoEVisionEncoder(FakeVisionCLIP(), training_stage=1)
    with torch.no_grad():
        stage1.budget.base_ratio_logits.add_(0.25)
        for parameter in stage1.sparse_pattern_generator.parameters():
            parameter.add_(0.1)
        for parameter in stage1.routers.parameters():
            parameter.fill_(7.0)
    checkpoint = {
        "encoder": {
            "budget": stage1.budget.state_dict(),
            "sparse_pattern_generator": stage1.sparse_pattern_generator.state_dict(),
        }
    }

    stage2 = SparMoEVisionEncoder(FakeVisionCLIP(), training_stage=2)
    router_before = {
        name: tensor.detach().clone() for name, tensor in stage2.routers.state_dict().items()
    }
    loaded_ratios = stage2.initialize_stage2_from_stage1(checkpoint)

    assert torch.equal(loaded_ratios, stage1.budget.base_ratios())
    for name, tensor in stage1.budget.state_dict().items():
        assert torch.equal(stage2.budget.state_dict()[name], tensor)
    for name, tensor in stage1.sparse_pattern_generator.state_dict().items():
        assert torch.equal(stage2.sparse_pattern_generator.state_dict()[name], tensor)
    for name, tensor in router_before.items():
        assert torch.equal(stage2.routers.state_dict()[name], tensor)
    assert_stage_parameter_contract(stage2, 2)


def test_stage1_spg_only_and_stage2_router_only_gradient_paths() -> None:
    torch.manual_seed(7)
    budget = LayerAdaptiveBudget(2, 0.65)
    spg = SparsePatternGenerator(
        num_layers=2,
        ffn_dims=16,
        embedding_dim=8,
        latent_dim=4,
        hyper_hidden_dim=4,
    )
    router = TokenCapacityRouter(model_dim=4)
    router.requires_grad_(False)

    retention = budget()
    patterns = spg.all_layers(retention)
    projection = torch.randn(16, 4)
    sparse_features = torch.stack([pattern.masks[-1] @ projection for pattern in patterns])
    dense_features = torch.randn_like(sparse_features)
    stage1_loss = Stage1SubspaceObjective(0.65)(
        sparse_features,
        dense_features,
        budget.base_ratios(),
        retention,
    )
    stage1_loss.total.backward()

    assert budget.base_ratio_logits.grad is not None
    assert torch.count_nonzero(budget.base_ratio_logits.grad)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in spg.parameters()
    )
    assert all(parameter.grad is None for parameter in router.parameters())

    budget.zero_grad(set_to_none=True)
    spg.zero_grad(set_to_none=True)
    budget.requires_grad_(False)
    spg.requires_grad_(False)
    router.requires_grad_(True)
    tokens = torch.randn(8, 4)
    routing = router(tokens)
    routing_input = LayerRoutingLossInput(
        logits=routing.logits,
        targets=torch.arange(8) % 4,
        gates=routing.gates,
    )
    frozen_retention = budget()
    frozen_patterns = spg.all_layers(frozen_retention)
    router_sparse_features = torch.stack(
        [pattern.masks[-1] @ projection for pattern in frozen_patterns]
    )
    stage2_loss = Stage2RouterObjective(0.65)(
        router_sparse_features,
        dense_features,
        [routing_input, routing_input],
        budget.base_ratios(),
        frozen_retention,
    )
    expected_total = 100.0 * stage2_loss.distillation + stage2_loss.routing
    assert torch.allclose(stage2_loss.total, expected_total)
    assert not stage2_loss.inherited_budget_error.requires_grad
    assert not stage2_loss.inherited_separation.requires_grad
    stage2_loss.total.backward()

    assert all(parameter.grad is None for parameter in budget.parameters())
    assert all(parameter.grad is None for parameter in spg.parameters())
    assert all(parameter.grad is not None for parameter in router.parameters())

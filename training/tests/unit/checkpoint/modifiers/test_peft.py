# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.modifiers.base import ModelModifier
from anemoi.training.checkpoint.modifiers.freezing import FreezingModifierStage
from anemoi.training.checkpoint.modifiers.peft import PeftModifierStage
from anemoi.training.checkpoint.pipeline import CheckpointPipeline

peft = pytest.importorskip("peft")

PROCESSOR_LINEARS = [
    "model.model.processor.0.lin_query",
    "model.model.processor.0.lin_value",
    "model.model.processor.0.mlp.0",
    "model.model.processor.0.mlp.2",
    "model.model.processor.1.lin_query",
    "model.model.processor.1.lin_value",
    "model.model.processor.1.mlp.0",
    "model.model.processor.1.mlp.2",
]
ALL_INNER_LINEARS = ["model.model.decoder.data", "model.model.encoder.data", *PROCESSOR_LINEARS]


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin_query = nn.Linear(8, 8)
        self.lin_value = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))


class GraphModel(nn.Module):
    """Stands in for the graph model at ``model.model``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleDict({"data": nn.Linear(4, 8)})
        self.processor = nn.ModuleList([Block(), Block()])
        self.decoder = nn.ModuleDict({"data": nn.Linear(8, 4)})


class Interface(nn.Module):
    """Stands in for ``AnemoiModelInterface``: owns the graph model plus parameters outside it."""

    def __init__(self) -> None:
        super().__init__()
        self.model = GraphModel()
        self.pre_processors = nn.Linear(4, 4)


class TrainingModule(nn.Module):
    """Stands in for the LightningModule: ``context.model``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Interface()


def lora(**options: object) -> dict:
    return {"r": 4, "lora_alpha": 8, "target_modules": "all-linear", **options}


def adapted_paths(module: nn.Module) -> list[str]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return sorted(name for name, m in module.named_modules() if isinstance(m, BaseTunerLayer))


def apply(stage: PeftModifierStage, model: nn.Module | None = None) -> CheckpointContext:
    return stage.process(CheckpointContext(model=model if model is not None else TrainingModule()))


def test_peft_stage_is_a_model_modifier() -> None:
    assert issubclass(PeftModifierStage, ModelModifier)


# --- build-time checks --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("modules_to_save", ["decoder"]),
        ("init_lora_weights", "pissa"),
        ("init_lora_weights", "olora"),
        ("init_lora_weights", False),
        ("bias", "all"),
        ("bias", "lora_only"),
        ("lora_bias", True),
        ("use_dora", True),
        ("layers_to_transform", [0]),
        ("rank_pattern", {"lin_query": 8}),
        ("alpha_pattern", {"lin_query": 16}),
        ("exclude_modules", ["lin_value"]),
    ],
)
def test_rejected_lora_option_raises_at_build(option: str, value: object) -> None:
    with pytest.raises(CheckpointConfigError, match=option):
        PeftModifierStage(peft_config=lora(**{option: value}))


@pytest.mark.parametrize("init", [True, "gaussian"])
def test_default_initialisations_are_accepted(init: object) -> None:
    PeftModifierStage(peft_config=lora(init_lora_weights=init))


def test_non_lora_config_is_rejected() -> None:
    with pytest.raises(CheckpointConfigError, match="LoraConfig"):
        PeftModifierStage(peft_config=peft.LoHaConfig(target_modules="all-linear"))


def test_lora_config_instance_is_accepted() -> None:
    PeftModifierStage(peft_config=peft.LoraConfig(r=4, target_modules="all-linear"))


def test_unknown_lora_option_is_rejected() -> None:
    with pytest.raises(CheckpointConfigError, match="rnak"):
        PeftModifierStage(peft_config=lora(rnak=4))


def test_missing_peft_raises_with_install_hint() -> None:
    with (
        patch.dict(sys.modules, {"peft": None}),
        pytest.raises(CheckpointConfigError, match=r"anemoi-training\[peft\]"),
    ):
        PeftModifierStage(peft_config=lora())


def run_config(dropout: float, gpus_per_model: int) -> object:
    return OmegaConf.create(
        {
            "system": {"hardware": {"num_gpus_per_model": gpus_per_model}},
            "training": {
                "checkpoint": {
                    "modifiers": [
                        {
                            "_target_": "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage",
                            "peft_config": lora(lora_dropout=dropout),
                        },
                    ],
                },
            },
        },
    )


def test_lora_dropout_with_model_sharding_raises_when_the_pipeline_is_built() -> None:
    # The builder owns this check: it reads system.hardware.num_gpus_per_model from the run
    # config, so the error comes before training starts, not at the first sharded forward.
    with pytest.raises(CheckpointConfigError, match="lora_dropout"):
        build_checkpoint_pipeline(run_config(dropout=0.1, gpus_per_model=2))


@pytest.mark.parametrize(("dropout", "gpus"), [(0.1, 1), (0.0, 2)])
def test_lora_dropout_allowed_without_sharding_or_at_zero(dropout: float, gpus: int) -> None:
    pipeline = build_checkpoint_pipeline(run_config(dropout=dropout, gpus_per_model=gpus))
    assert [type(stage) for stage in pipeline.stages] == [PeftModifierStage]


# --- target resolution ------------------------------------------------------------------


def test_all_linear_adapts_every_linear_in_the_graph_model_only() -> None:
    result = apply(PeftModifierStage(peft_config=lora()))
    assert adapted_paths(result.model) == ALL_INNER_LINEARS
    assert not isinstance(result.model.model.pre_processors, peft.tuners.lora.LoraLayer)


def test_exact_full_paths_adapt_exactly_those_modules() -> None:
    targets = ["model.model.processor.0.lin_query", "model.model.decoder.data"]
    result = apply(PeftModifierStage(peft_config=lora(target_modules=targets)))
    assert adapted_paths(result.model) == sorted(targets)


def test_regex_is_fullmatched_against_full_paths() -> None:
    regex = r"model\.model\.processor\.\d+\.lin_(query|value)"
    result = apply(PeftModifierStage(peft_config=lora(target_modules=regex)))
    assert adapted_paths(result.model) == [p for p in PROCESSOR_LINEARS if "lin_" in p]


def test_suffix_only_name_is_not_a_full_path() -> None:
    with pytest.raises(CheckpointConfigError, match="lin_query"):
        apply(PeftModifierStage(peft_config=lora(target_modules=["lin_query"])))


def test_path_outside_the_graph_model_raises() -> None:
    with pytest.raises(CheckpointConfigError, match=re.escape("model.pre_processors")):
        apply(PeftModifierStage(peft_config=lora(target_modules=["model.pre_processors"])))


def test_regex_matching_nothing_raises() -> None:
    with pytest.raises(CheckpointConfigError, match="matched no"):
        apply(PeftModifierStage(peft_config=lora(target_modules=r"model\.model\.nowhere\..*")))


def test_target_that_is_not_a_linear_layer_raises() -> None:
    with pytest.raises(CheckpointConfigError, match=re.escape("model.model.processor.0.norm")):
        apply(PeftModifierStage(peft_config=lora(target_modules=["model.model.processor.0.norm"])))


def test_missing_target_modules_raises() -> None:
    with pytest.raises(CheckpointConfigError, match="target_modules"):
        apply(PeftModifierStage(peft_config={"r": 4}))


def test_model_without_graph_model_raises() -> None:
    with pytest.raises(CheckpointConfigError, match=re.escape("model.model")):
        apply(PeftModifierStage(peft_config=lora()), model=nn.Linear(4, 4))


def test_model_that_already_has_adapters_raises() -> None:
    model = apply(PeftModifierStage(peft_config=lora())).model
    with pytest.raises(CheckpointConfigError, match="already"):
        apply(PeftModifierStage(peft_config=lora()), model=model)


# --- what trains ---------------------------------------------------------------------------


def trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_only_adapters_train_by_default() -> None:
    result = apply(PeftModifierStage(peft_config=lora()))
    names = trainable(result.model)
    assert names
    assert all(".lora_A." in n or ".lora_B." in n for n in names)
    assert not any(n.startswith("model.pre_processors") for n in names)


def test_trainable_modules_use_full_paths_and_train() -> None:
    stage = PeftModifierStage(
        peft_config=lora(target_modules=r"model\.model\.processor\..*lin_query"),
        trainable_modules=["model.model.decoder", "model.pre_processors"],
    )
    names = trainable(apply(stage).model)
    assert {n for n in names if ".lora_" not in n} == {
        "model.model.decoder.data.weight",
        "model.model.decoder.data.bias",
        "model.pre_processors.weight",
        "model.pre_processors.bias",
    }


def test_unknown_trainable_module_raises() -> None:
    stage = PeftModifierStage(peft_config=lora(), trainable_modules=["model.model.head"])
    with pytest.raises(CheckpointConfigError, match=re.escape("model.model.head")):
        apply(stage)


def test_adapted_layer_output_is_unchanged_at_injection() -> None:
    model = TrainingModule()
    x = torch.randn(3, 8)
    before = model.model.model.processor[0].lin_query(x)
    after = apply(PeftModifierStage(peft_config=lora()), model=model).model.model.model.processor[0].lin_query(x)
    torch.testing.assert_close(after, before)


def test_metadata_records_the_adapters() -> None:
    result = apply(PeftModifierStage(peft_config=lora(target_modules=["model.model.decoder.data"])))
    (record,) = result.metadata["modifiers_applied"]
    assert record["type"] == "peft"
    assert record["adapted_modules"] == ["model.model.decoder.data"]


# --- composition --------------------------------------------------------------------------


def test_freezing_after_peft_is_rejected() -> None:
    peft_stage = PeftModifierStage(peft_config=lora())
    freezing = FreezingModifierStage(submodules_to_freeze=["model.model.encoder"])
    with pytest.raises(CheckpointConfigError, match="FreezingModifierStage"):
        CheckpointPipeline(stages=[peft_stage, freezing])


def test_freezing_before_peft_is_accepted() -> None:
    freezing = FreezingModifierStage(submodules_to_freeze=["model.model.encoder"])
    CheckpointPipeline(stages=[freezing, PeftModifierStage(peft_config=lora())])


def test_stage_builds_from_a_hydra_config() -> None:
    config = OmegaConf.create(
        {
            "_target_": "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage",
            "peft_config": {"r": 4, "lora_alpha": 8, "target_modules": ["model.model.decoder.data"]},
            "trainable_modules": ["model.model.encoder"],
        },
    )
    pipeline = CheckpointPipeline(stages=[config])
    result = pipeline.execute(CheckpointContext(model=TrainingModule()))
    assert adapted_paths(result.model) == ["model.model.decoder.data"]


def test_all_linear_on_a_real_processor() -> None:
    from anemoi.models.layers.processor import TransformerProcessor
    from anemoi.models.layers.utils import load_layer_kernels

    processor = TransformerProcessor(
        num_layers=2,
        num_channels=16,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        dropout_p=0.0,
        attention_implementation="scaled_dot_product_attention",
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=load_layer_kernels(instance=False),
    )
    model = TrainingModule()
    model.model.model.processor = processor
    expected = sorted(
        f"model.model.processor.{name}" for name, m in processor.named_modules() if isinstance(m, nn.Linear)
    )
    result = apply(PeftModifierStage(peft_config=lora(target_modules=r"model\.model\.processor\..*")), model=model)
    assert expected
    assert [p for p in adapted_paths(result.model) if p.startswith("model.model.processor")] == expected

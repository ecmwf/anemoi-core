# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import sys
from collections.abc import Callable
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
from anemoi.training.checkpoint.modifiers.peft import adapter_record
from anemoi.training.checkpoint.pipeline import CheckpointPipeline

peft = pytest.importorskip("peft")

PROCESSOR_LINEARS = [
    "processor.0.lin_query",
    "processor.0.lin_value",
    "processor.0.mlp.0",
    "processor.0.mlp.2",
    "processor.1.lin_query",
    "processor.1.lin_value",
    "processor.1.mlp.0",
    "processor.1.mlp.2",
]
ALL_LINEARS = sorted(["decoder.data", "encoder.data", *PROCESSOR_LINEARS])
# A PEFT regex is matched in full against module names in the graph model.
PROCESSOR_REGEX = r"processor\.\d+\.(lin_query|lin_value|mlp\.0|mlp\.2)"


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


# The PEFT methods the tests run against: each builds a config from keyword options.
METHODS: dict[str, Callable[..., object]] = {
    "lora": lambda **options: peft.LoraConfig(**{"r": 2, "lora_alpha": 4, "target_modules": "all-linear", **options}),
    "miss": lambda **options: peft.MissConfig(**{"r": 2, "target_modules": "all-linear", **options}),
    "loha": lambda **options: peft.LoHaConfig(**{"r": 2, "alpha": 4, "target_modules": "all-linear", **options}),
}


def lora(**options: object) -> object:
    return METHODS["lora"](**options)


def adapted(model: nn.Module) -> list[str]:
    """Names, relative to the graph model, of the modules holding adapters or a trained copy."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    graph_model = model.model.model
    return sorted(
        name for name, m in graph_model.named_modules() if isinstance(m, BaseTunerLayer | ModulesToSaveWrapper)
    )


def trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def apply(stage: PeftModifierStage, model: nn.Module | None = None) -> CheckpointContext:
    return stage.process(CheckpointContext(model=model if model is not None else TrainingModule()))


def test_peft_stage_is_a_model_modifier() -> None:
    assert issubclass(PeftModifierStage, ModelModifier)


# --- build-time checks --------------------------------------------------------------------


def test_missing_peft_raises_with_install_hint() -> None:
    config = lora()
    with (
        patch.dict(sys.modules, {"peft": None}),
        pytest.raises(CheckpointConfigError, match=r"anemoi-training\[peft\]"),
    ):
        PeftModifierStage(peft_config=config)


def test_missing_peft_in_a_config_raises_with_install_hint() -> None:
    # From a config, Hydra imports the peft_config target before the stage exists.
    stage_config = {
        "_target_": "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage",
        "peft_config": {"_target_": "peft.LoraConfig", "r": 2, "target_modules": "all-linear"},
    }
    with (
        patch.dict(sys.modules, {"peft": None}),
        pytest.raises(CheckpointConfigError, match=r"needs the peft package.*anemoi-training\[peft\]"),
    ):
        CheckpointPipeline([stage_config])


def test_a_mapping_is_not_a_peft_config() -> None:
    with pytest.raises(CheckpointConfigError, match="_target_"):
        PeftModifierStage(peft_config={"r": 4, "target_modules": "all-linear"})


def test_prompt_learning_is_refused() -> None:
    with pytest.raises(CheckpointConfigError, match="prompt-learning"):
        PeftModifierStage(peft_config=peft.PromptTuningConfig(num_virtual_tokens=4))


@pytest.mark.parametrize(
    ("config", "warned"),
    [
        (lora(bias="all"), "bias='all'"),
        (lora(lora_bias=True), "lora_bias=True"),
        (lora(modules_to_save=["decoder.data"]), "modules_to_save=['decoder.data']"),
        (lora(init_lora_weights="pissa"), "init_lora_weights='pissa'"),
        (lora(init_lora_weights=False), "init_lora_weights=False"),
        (METHODS["miss"](modules_to_save=["decoder.data"]), "modules_to_save"),
    ],
)
def test_options_beyond_the_adapters_warn_and_are_kept(
    config: object,
    warned: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        stage = PeftModifierStage(peft_config=config)
    assert warned in caplog.text
    assert stage.peft_config.to_dict() == config.to_dict()


@pytest.mark.parametrize("method", sorted(METHODS))
def test_default_options_do_not_warn(method: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        PeftModifierStage(peft_config=METHODS[method]())
    assert not caplog.records


def run_config(peft_config: dict, gpus_per_model: int) -> object:
    return OmegaConf.create(
        {
            "system": {"hardware": {"num_gpus_per_model": gpus_per_model}},
            "training": {
                "checkpoint": {
                    "modifiers": [
                        {
                            "_target_": "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage",
                            "peft_config": peft_config,
                        },
                    ],
                },
            },
        },
    )


LORA = {"_target_": "peft.LoraConfig", "r": 2, "target_modules": "all-linear"}
MISS = {"_target_": "peft.MissConfig", "r": 2, "target_modules": "all-linear"}
LOHA = {"_target_": "peft.LoHaConfig", "r": 2, "target_modules": "all-linear"}


@pytest.mark.parametrize(
    ("peft_config", "option"),
    [
        ({**LORA, "lora_dropout": 0.1}, "lora_dropout"),
        ({**MISS, "miss_dropout": 0.1}, "miss_dropout"),
        ({**LOHA, "module_dropout": 0.1}, "module_dropout"),
    ],
)
def test_adapter_dropout_with_model_sharding_raises_when_the_pipeline_is_built(peft_config: dict, option: str) -> None:
    # The builder owns this check: it reads system.hardware.num_gpus_per_model from the run
    # config, so the error comes before training starts, not at the first sharded forward.
    with pytest.raises(CheckpointConfigError, match=option):
        build_checkpoint_pipeline(run_config(peft_config, gpus_per_model=2))


@pytest.mark.parametrize(
    ("peft_config", "gpus"),
    [({**LORA, "lora_dropout": 0.1}, 1), ({**LORA, "lora_dropout": 0.0}, 2), (MISS, 2)],
)
def test_adapter_dropout_allowed_without_sharding_or_at_zero(peft_config: dict, gpus: int) -> None:
    pipeline = build_checkpoint_pipeline(run_config(peft_config, gpus_per_model=gpus))
    assert [type(stage) for stage in pipeline.stages] == [PeftModifierStage]


# --- what PEFT adapts and trains ----------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_linear_adapts_every_linear_in_the_graph_model_only(method: str) -> None:
    result = apply(PeftModifierStage(peft_config=METHODS[method]()))
    assert adapted(result.model) == ALL_LINEARS
    assert type(result.model.model.pre_processors) is nn.Linear


@pytest.mark.parametrize("method", sorted(METHODS))
def test_only_the_adapters_of_the_graph_model_train(method: str) -> None:
    model = apply(PeftModifierStage(peft_config=METHODS[method]())).model
    in_graph_model = {name for name in trainable(model) if name.startswith("model.model.")}
    assert in_graph_model
    assert all(".default" in name and ".base_layer." not in name for name in in_graph_model)
    # Outside the graph model PEFT changes nothing.
    assert {name for name in trainable(model) if name.startswith("model.pre_processors")} == {
        "model.pre_processors.weight",
        "model.pre_processors.bias",
    }


@pytest.mark.parametrize("method", sorted(METHODS))
def test_adapted_layer_output_is_unchanged_at_injection(method: str) -> None:
    model = TrainingModule()
    x = torch.randn(3, 8)
    before = model.model.model.processor[0].lin_query(x)
    apply(PeftModifierStage(peft_config=METHODS[method]()), model=model)
    torch.testing.assert_close(model.model.model.processor[0].lin_query(x), before)


@pytest.mark.parametrize(
    ("target_modules", "expected"),
    [
        (["lin_query"], ["processor.0.lin_query", "processor.1.lin_query"]),  # PEFT matches names by suffix
        (["processor.0.lin_query", "decoder.data"], ["decoder.data", "processor.0.lin_query"]),
        (r"processor\.\d+\.lin_(query|value)", [p for p in PROCESSOR_LINEARS if "lin_" in p]),
        (PROCESSOR_REGEX, PROCESSOR_LINEARS),
    ],
)
def test_target_modules_follow_peft_matching_within_the_graph_model(
    target_modules: object,
    expected: list[str],
) -> None:
    result = apply(PeftModifierStage(peft_config=lora(target_modules=target_modules)))
    assert adapted(result.model) == expected


def test_target_matching_nothing_raises() -> None:
    with pytest.raises(CheckpointConfigError, match="nowhere"):
        apply(PeftModifierStage(peft_config=lora(target_modules=["nowhere"])))


def test_target_peft_cannot_adapt_raises() -> None:
    with pytest.raises(CheckpointConfigError, match="not supported"):
        apply(PeftModifierStage(peft_config=lora(target_modules=["processor.0.norm"])))


def test_model_without_graph_model_raises() -> None:
    with pytest.raises(CheckpointConfigError, match=r"'model\.model'"):
        apply(PeftModifierStage(peft_config=lora()), model=nn.Linear(4, 4))


def test_bias_all_trains_the_base_biases() -> None:
    model = apply(PeftModifierStage(peft_config=lora(target_modules=["decoder.data"], bias="all"))).model
    assert "model.model.decoder.data.base_layer.bias" in trainable(model)
    assert "model.model.processor.0.lin_query.bias" in trainable(model)
    assert "model.model.decoder.data.base_layer.weight" not in trainable(model)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_modules_to_save_train_a_copy_of_the_module(method: str) -> None:
    config = METHODS[method](target_modules=PROCESSOR_REGEX, modules_to_save=["decoder.data"])
    model = apply(PeftModifierStage(peft_config=config)).model
    assert "decoder.data" in adapted(model)
    names = trainable(model)
    assert {"model.model.decoder.data.modules_to_save.default.weight"} <= names
    assert not any(".original_module." in name for name in names)


def test_model_that_already_has_adapters_is_merged_first() -> None:
    model = apply(PeftModifierStage(peft_config=METHODS["miss"]())).model
    result = apply(PeftModifierStage(peft_config=lora(target_modules=["decoder.data"])), model=model)
    assert adapted(result.model) == ["decoder.data"]
    assert result.metadata["modifiers_applied"][-1]["merged_previous_adapters"] is True


def test_metadata_records_the_adapters() -> None:
    result = apply(PeftModifierStage(peft_config=METHODS["miss"](target_modules=["decoder.data"])))
    (entry,) = result.metadata["modifiers_applied"]
    assert entry["type"] == "peft"
    assert entry["peft_type"] == "MISS"
    assert entry["adapted_modules"] == ["decoder.data"]


# --- composition --------------------------------------------------------------------------


def test_freezing_after_peft_freezes_the_named_adapters() -> None:
    peft_stage = PeftModifierStage(peft_config=lora(target_modules=PROCESSOR_REGEX))
    freezing = FreezingModifierStage(submodules_to_freeze=["model.model.processor.0"])
    result = CheckpointPipeline(stages=[peft_stage, freezing]).execute(CheckpointContext(model=TrainingModule()))

    names = trainable(result.model)
    assert not any(name.startswith("model.model.processor.0.") for name in names)
    assert "model.model.processor.1.lin_query.lora_A.default.weight" in names


def test_freezing_before_peft_is_accepted() -> None:
    freezing = FreezingModifierStage(submodules_to_freeze=["model.model.encoder"])
    CheckpointPipeline(stages=[freezing, PeftModifierStage(peft_config=lora())])


def test_stage_builds_from_a_hydra_config_with_any_method() -> None:
    config = OmegaConf.create(
        {
            "_target_": "anemoi.training.checkpoint.modifiers.peft.PeftModifierStage",
            "peft_config": {
                "_target_": "peft.MissConfig",
                "r": 2,
                "target_modules": ["decoder.data"],
                "modules_to_save": ["encoder.data"],
            },
        },
    )
    (stage,) = CheckpointPipeline(stages=[config]).stages
    assert isinstance(stage.peft_config, peft.MissConfig)

    result = stage.process(CheckpointContext(model=TrainingModule()))

    assert adapted(result.model) == ["decoder.data", "encoder.data"]
    json.dumps(adapter_record(result.model))  # no ListConfig left over from Hydra


def test_peft_preset_builds_and_adapts_the_graph_model() -> None:
    """The shipped preset instantiates as written and adapts every Linear in the graph model."""
    from importlib.resources import files

    preset = files("anemoi.training") / "config" / "training" / "checkpoint" / "modifiers" / "peft.yaml"
    (stage_cfg,) = OmegaConf.create(preset.read_text())
    assert set(stage_cfg) == {"_target_", "peft_config"}

    (stage,) = CheckpointPipeline(stages=[stage_cfg]).stages
    result = stage.process(CheckpointContext(model=TrainingModule()))

    assert isinstance(stage.peft_config, peft.LoraConfig)
    assert adapted(result.model) == ALL_LINEARS


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
    expected = sorted(f"processor.{name}" for name, m in processor.named_modules() if isinstance(m, nn.Linear))
    result = apply(PeftModifierStage(peft_config=lora()), model=model)
    assert expected
    assert [name for name in adapted(result.model) if name.startswith("processor.")] == expected

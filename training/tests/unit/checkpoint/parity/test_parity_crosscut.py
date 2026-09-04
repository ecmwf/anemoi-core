# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cross-cutting parity tests for the checkpoint pipeline.

These close genuine coverage gaps that span more than one layer of the
checkpoint pipeline: cross-dtype loading, load determinism / idempotency,
end-to-end metadata preservation, the deprecated-key migration advice, and
per-preset config composition of the ``training.checkpoint`` surface.

Everything here is CPU-only and deterministic. Load-probe tests build a small
``nn.Module`` plus a synthetic checkpoint, construct a real
:class:`CheckpointContext`, and drive the real loading strategy / source /
builder — no mocking of the code under test.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Never
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader
from anemoi.training.checkpoint.sources.base import ResolveOnlySource
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.schemas.base_schema import _DEPRECATED_KEYS
from anemoi.training.schemas.training import CheckpointPipelineSchema
from anemoi.training.tasks.forecaster import Forecaster
from anemoi.training.train.methods.base import BaseTrainingModule

if TYPE_CHECKING:
    from pathlib import Path

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOCAL_SOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"
_WEIGHTS_ONLY = "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"
_WARM_START = "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"
_TRANSFER_LEARNING = "anemoi.training.checkpoint.loading.strategies.TransferLearningLoader"
_FREEZING = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"


class _SmallNet(nn.Module):
    """Minimal two-layer CPU model for load probes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _index_collection(name_to_index: dict[str, int]) -> object:
    """Build an object exposing ``.name_to_index`` like a real IndexCollection."""
    return type("IndexCollection", (), {"name_to_index": name_to_index})()


class _PicklableIndex:
    """An ``IndexCollection`` stand-in that survives ``torch.save`` (module-level, so it pickles)."""

    def __init__(self, name_to_index: dict[str, int]) -> None:
        self.name_to_index = name_to_index


# --- mixed precision / dtype load (fp32 <-> bf16) ---------------------


def test_weights_only_casts_fp32_state_dict_into_bfloat16_model() -> None:
    """Loading an fp32 state dict into a bf16 model casts to the model dtype, no corruption."""
    torch.manual_seed(0)
    source = _SmallNet()
    source_state = {key: value.clone() for key, value in source.state_dict().items()}

    target = _SmallNet().to(torch.bfloat16)
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    loaded = target.state_dict()
    for key, source_value in source_state.items():
        assert loaded[key].dtype == torch.bfloat16, f"{key} lost the model dtype"
        # Values equal the fp32 source cast down to bf16 — the only lossy step is the
        # documented cast, never a silent reinterpret of the raw bytes.
        assert torch.equal(loaded[key], source_value.to(torch.bfloat16))


def test_weights_only_casts_bfloat16_state_dict_into_fp32_model() -> None:
    """Loading a bf16 state dict into an fp32 model widens to fp32 with matching values."""
    torch.manual_seed(0)
    source = _SmallNet().to(torch.bfloat16)
    source_state = {key: value.clone() for key, value in source.state_dict().items()}

    target = _SmallNet()  # fp32
    context = CheckpointContext(model=target, checkpoint_data={"state_dict": source_state})

    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    loaded = target.state_dict()
    for key, source_value in source_state.items():
        assert loaded[key].dtype == torch.float32, f"{key} lost the model dtype"
        assert torch.equal(loaded[key], source_value.to(torch.float32))


# --- pipeline determinism (two identical load-probes match) -----------


def test_same_checkpoint_loads_identically_into_two_models(tmp_path: Path) -> None:
    """The same checkpoint loaded into two fresh models yields bit-exact state dicts."""
    torch.manual_seed(0)
    source = _SmallNet()
    ckpt_path = tmp_path / "checkpoint.ckpt"
    torch.save({"state_dict": source.state_dict()}, ckpt_path)

    # Two models built from DIFFERENT initial weights: if loading failed to overwrite
    # every slot, the post-load state dicts would diverge and the test would fail.
    torch.manual_seed(1)
    model_a = _SmallNet()
    torch.manual_seed(2)
    model_b = _SmallNet()
    assert not torch.equal(model_a.state_dict()["fc1.weight"], model_b.state_dict()["fc1.weight"])

    for model in (model_a, model_b):
        context = CheckpointContext(checkpoint_path=ckpt_path, model=model)
        asyncio.run(LocalSource().process(context))
        asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    source_state = source.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        # atol=0: identical bytes, not merely close.
        assert torch.equal(state_a[key], state_b[key]), f"{key} differs between loads"
        assert torch.equal(state_a[key], source_state[key]), f"{key} does not match the source"


# --- a resume is one load: Lightning's, corrected by the hook -----------


class _StepStub(BaseTrainingModule):
    """Concrete BaseTrainingModule whose training step is unused in these tests."""

    def __init__(self) -> None:
        pass

    def _step(self, batch: object, validation_mode: bool = False) -> Never:
        raise NotImplementedError


class _BufferProcessor(nn.Module):
    """A processor whose only state is one buffer, so refreshes are visible in the state dict."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor([value], dtype=torch.float32))


class _TendencyModel(nn.Module):
    """A model with state and tendency processors, keyed the way the trainer's inner model is."""

    def __init__(self, lead_times: list[str], offset: float) -> None:
        super().__init__()
        self.pre_processors = nn.ModuleDict({"data": Processors([["dummy", _BufferProcessor(offset)]])})
        self.post_processors = nn.ModuleDict(
            {"data": Processors([["dummy", _BufferProcessor(offset + 100)]], inverse=True)},
        )
        pre_tend = StepwiseProcessors(lead_times)
        post_tend = StepwiseProcessors(lead_times)
        for idx, lead_time in enumerate(lead_times):
            pre_tend.set(lead_time, Processors([["dummy", _BufferProcessor(offset + idx)]]))
            post_tend.set(lead_time, Processors([["dummy", _BufferProcessor(offset + idx + 50)]], inverse=True))
        self.pre_processors_tendencies = nn.ModuleDict({"data": pre_tend})
        self.post_processors_tendencies = nn.ModuleDict({"data": post_tend})
        self.body = nn.Linear(2, 2)


def _lightning_module_wrapping(model: nn.Module, *, tendencies: bool = False) -> _StepStub:
    """Build a minimal training module whose ``.model`` is ``model``."""
    from types import SimpleNamespace

    module = _StepStub.__new__(_StepStub)
    nn.Module.__init__(module)
    module.task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module.model = model
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=SimpleNamespace(states=False, tendencies=tendencies)),
    )
    return module


def _legacy_on_load_checkpoint(module: _StepStub, checkpoint: dict) -> None:
    """``BaseTrainingModule.on_load_checkpoint`` as it stands at origin/main 4842e8bb8.

    Reproduced from that revision's ``train/methods/base.py`` (edge-perm migration,
    then ``_update_checkpoint_state_dict_for_load``, then ``_ckpt_model_name_to_index``,
    then ``_ckpt_variables_metadata``), minus the task-state and datamodule steps that
    need a trainer. This is the reference a resume must match bit for bit. Note the
    edge-perm migration is a no-op for a model without a ``StaticGraphProvider``, so
    what this pins is the processor refresh and the metadata restoration.
    """
    import importlib

    from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint

    edge_perm = importlib.import_module("anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix").migrate
    edge_perm(checkpoint, model=module)

    update_cfg = module.config.training.update_ds_stats_on_ckpt_load
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict) and (update_cfg.states or update_cfg.tendencies):
        processor_prefixes: tuple[str, ...] = ()
        if update_cfg.states:
            processor_prefixes += ("model.pre_processors.", "model.post_processors.")
        if update_cfg.tendencies:
            processor_prefixes += ("model.pre_processors_tendencies.", "model.post_processors_tendencies.")
        for key in list(state_dict.keys()):
            if key.startswith(processor_prefixes):
                del state_dict[key]
        model_state_dict = module.model.state_dict()
        processor_prefixes += tuple(f"model.{k}" for k in model_state_dict if "model_output_idx" in k)
        for key, value in model_state_dict.items():
            full_key = f"model.{key}"
            if full_key.startswith(processor_prefixes):
                state_dict[full_key] = value

    module._ckpt_model_name_to_index = {
        dataset_name: data_indices.name_to_index
        for dataset_name, data_indices in checkpoint["hyper_parameters"]["data_indices"].items()
    }
    module._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
        checkpoint,
        module._ckpt_model_name_to_index,
    )


def test_resume_is_one_lightning_load_and_matches_the_legacy_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume performs no pipeline load; the hook corrects the dict Lightning loads.

    With ``update_ds_stats_on_ckpt_load.tendencies=True`` the tendency processors must
    come from the live model, not the checkpoint. The pipeline only resolves the file
    (``torch.load`` is never called), the model keeps its own weights until ``fit()``,
    and the hook applies ``apply_checkpoint_corrections`` to Lightning's dict. The
    result is compared bit for bit against the legacy hook at origin/main 4842e8bb8.
    """
    old_model = _TendencyModel(["6h", "12h", "18h"], offset=10.0)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": _PicklableIndex({"t2m": 0})}},
    }
    ckpt_path = tmp_path / "run_A" / "last.ckpt"
    ckpt_path.parent.mkdir()
    torch.save(checkpoint, ckpt_path)

    real_load = torch.load
    loads: list[object] = []

    def _spy_load(*args: object, **kwargs: object) -> object:
        loads.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)

    module = _lightning_module_wrapping(_TendencyModel(["6h", "12h"], offset=1.0), tendencies=True)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {"_target_": _LOCAL_SOURCE, "path": str(ckpt_path)},
                    "loading": {"_target_": _WARM_START},
                },
                "update_ds_stats_on_ckpt_load": {"states": False, "tendencies": True},
            },
        },
    )

    executed = asyncio.run(build_checkpoint_pipeline(cfg).execute(CheckpointContext(model=module, config=cfg)))

    # No pipeline load: the file was only resolved, the model is untouched.
    assert loads == []
    assert executed.checkpoint_path == ckpt_path.resolve()
    assert executed.checkpoint_data is None
    assert not getattr(module, "weights_initialized", False)
    for key, value in before.items():
        assert torch.equal(module.state_dict()[key], value)

    # Lightning's ckpt_path load: read the file, run the hook on that dict, load it.
    lightning_dict = real_load(executed.checkpoint_path, weights_only=False, map_location="cpu")
    BaseTrainingModule.on_load_checkpoint(module, lightning_dict)
    module.load_state_dict(lightning_dict["state_dict"], strict=True)

    # The legacy path, on an identical module and a fresh read of the same file.
    reference = _lightning_module_wrapping(_TendencyModel(["6h", "12h"], offset=1.0), tendencies=True)
    legacy_dict = real_load(ckpt_path, weights_only=False, map_location="cpu")
    _legacy_on_load_checkpoint(reference, legacy_dict)
    reference.load_state_dict(legacy_dict["state_dict"], strict=True)

    resumed = module.state_dict()
    for key, value in reference.state_dict().items():
        assert torch.equal(resumed[key], value), f"{key} differs from the legacy hook path"
    # And the tendency processors really are the live model's, not the checkpoint's stale 10.x.
    assert torch.equal(
        resumed["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
        torch.tensor([1.0]),
    )
    assert module._ckpt_model_name_to_index == {"data": {"t2m": 0}}
    assert module._ckpt_variables_metadata == reference._ckpt_variables_metadata


# --- metadata round-trip through a weights-only load ------------------


def test_weights_only_preserves_name_index_variables_metadata_and_hyper_parameters() -> None:
    """A weights-only load preserves name_to_index, variables_metadata and hyper_parameters."""
    torch.manual_seed(0)
    name_to_index = {"t2m": 0, "u10": 1, "v10": 2}
    variables_metadata = {"t2m": {"units": "K"}, "u10": {"units": "m s**-1"}, "v10": {"units": "m s**-1"}}
    hyper_parameters = {
        "data_indices": {"era5": _index_collection(name_to_index)},
        "metadata": {"dataset": {"era5": {"variables_metadata": variables_metadata}}},
    }
    model = _SmallNet()
    checkpoint_data = {
        "state_dict": {key: torch.randn_like(value) for key, value in model.state_dict().items()},
        "hyper_parameters": hyper_parameters,
    }

    context = CheckpointContext(model=model, checkpoint_data=checkpoint_data)
    asyncio.run(WeightsOnlyLoader(strict=True).process(context))

    assert model._ckpt_model_name_to_index == {"era5": name_to_index}
    assert model._ckpt_variables_metadata == {"era5": variables_metadata}
    # hyper_parameters must survive the load untouched (same object, equal content).
    assert context.checkpoint_data["hyper_parameters"] is hyper_parameters
    assert (
        context.checkpoint_data["hyper_parameters"]["metadata"]["dataset"]["era5"]["variables_metadata"]
        == variables_metadata
    )


# --- the deprecated-key migration advice actually builds a pipeline ---


def _replacement_checkpoint_config(deprecated_key: str) -> dict:
    """The ``training.checkpoint`` block the hint for ``deprecated_key`` recommends."""
    return {
        "training.run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": False}},
        "training.fork_run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": True}},
        "system.input.warm_start": {"source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"}},
        "training.load_weights_only": {
            "source": {"_target_": _RUN_SOURCE, "run_id": "abc"},
            "loading": {"_target_": _WEIGHTS_ONLY},
        },
        "training.transfer_learning": {
            "source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"},
            "loading": {"_target_": _TRANSFER_LEARNING, "skip_mismatched": True},
        },
        "training.submodules_to_freeze": {
            "modifiers": [{"_target_": _FREEZING, "submodules_to_freeze": ["encoder"]}],
        },
    }[deprecated_key]


def test_replacement_config_map_covers_every_deprecated_key() -> None:
    """The advice-under-test map stays in lockstep with the real _DEPRECATED_KEYS registry."""
    covered = {
        "training.run_id",
        "training.fork_run_id",
        "system.input.warm_start",
        "training.load_weights_only",
        "training.transfer_learning",
        "training.submodules_to_freeze",
    }
    assert covered == set(_DEPRECATED_KEYS)


@pytest.mark.parametrize("deprecated_key", sorted(_DEPRECATED_KEYS))
def test_deprecated_key_replacement_config_builds_and_validates(deprecated_key: str) -> None:
    """Each removed key's recommended replacement composes into a valid, buildable pipeline."""
    if deprecated_key == "training.submodules_to_freeze":
        pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")

    checkpoint_block = _replacement_checkpoint_config(deprecated_key)

    # The schema that governs this surface must accept the advice.
    CheckpointPipelineSchema(**checkpoint_block)

    cfg = OmegaConf.create({"training": {"checkpoint": checkpoint_block}})
    pipeline = build_checkpoint_pipeline(cfg)
    stage_names = [type(stage).__name__ for stage in pipeline.stages]

    if "source" in checkpoint_block:
        assert stage_names[0].endswith("Source")
    if "loading" in checkpoint_block:
        assert any(name.endswith("Loader") for name in stage_names)
    else:
        # Source without a loading block is a resume: the source runs resolve-only and
        # Trainer.fit(ckpt_path=) performs the load, as the removed keys used to.
        assert not any(name.endswith("Loader") for name in stage_names)
    if "modifiers" in checkpoint_block:
        assert any(name.endswith("Stage") for name in stage_names)

    # Non-tautological attribute checks: the advice configures the intended behaviour.
    if deprecated_key == "training.run_id":
        source = pipeline.stages[0].source
        assert source.run_id == "abc"
        assert source.fork is False
    elif deprecated_key == "training.fork_run_id":
        source = pipeline.stages[0].source
        assert source.run_id == "abc"
        assert source.fork is True
    elif deprecated_key == "system.input.warm_start":
        assert isinstance(pipeline.stages[0], ResolveOnlySource)
        assert isinstance(pipeline.stages[0].source, LocalSource)
    elif deprecated_key == "training.transfer_learning":
        loader = pipeline.stages[1]
        assert type(loader).__name__ == "TransferLearningLoader"
        assert loader.skip_mismatched is True
    elif deprecated_key == "training.submodules_to_freeze":
        assert pipeline.stages[0].submodules_to_freeze == ["encoder"]


# --- each preset composes with a checkpoint.source overlay ------------

_PRESETS = ["config", "lam", "multi", "stretched", "ensemble_crps"]


@pytest.mark.parametrize("preset", _PRESETS)
def test_preset_composes_with_checkpoint_source_overlay(preset: str) -> None:
    """Adding a RunIdSource overlay to each preset composes cleanly and yields a valid pipeline.

    This exercises the composed ``training.checkpoint`` surface per preset (no defaults
    conflict from the ``training/checkpoint/source`` group) and validates it with the
    schema and builder that own it.
    """
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        cfg = compose(
            config_name=preset,
            overrides=[
                "training/checkpoint/source=run",
                "+training.checkpoint.source.run_id=abc123",
            ],
        )

    checkpoint_block = OmegaConf.to_container(cfg.training.checkpoint, resolve=True)
    CheckpointPipelineSchema(**checkpoint_block)

    pipeline = build_checkpoint_pipeline(cfg)
    # Source only: a resume, so the RunIdSource runs resolve-only for Trainer.fit(ckpt_path=).
    assert isinstance(pipeline.stages[0], ResolveOnlySource)
    assert type(pipeline.stages[0].source).__name__ == "RunIdSource"
    assert pipeline.stages[0].source.run_id == "abc123"


# --- the resume hook, like the pipeline, never runs chunking_fix on autoencoder weights ---


@pytest.fixture
def spy_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Observe ``chunking_fix`` at its resolver, leaving the real migration ledger reachable."""
    spy = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})
    monkeypatch.setattr(
        "anemoi.training.checkpoint.loading.base._load_chunking_fix_migration",
        lambda: spy,
    )
    return spy


def _ledger(*names: str) -> list[dict]:
    """A checkpoint migration ledger recording ``names`` as already applied."""
    from anemoi.models.migrations.migrator import MigrationMetadata

    return [
        {
            "name": name,
            "metadata": MigrationMetadata(versions={"migration": "1.0.0", "anemoi-models": "0.11.0"}),
            "signature": f"signature-of-{name}",
        }
        for name in names
    ]


def test_resume_hook_does_not_migrate_an_autoencoder_checkpoint_behind_the_ledger(
    caplog: pytest.LogCaptureFixture,
    spy_chunking_migration: MagicMock,
) -> None:
    """On a resume the hook is the only correction pass, so the autoencoder case is pinned there too.

    Same checkpoint shape as the pipeline-path test in ``test_format_migrations.py``: a
    ``NoOpProcessor`` config and a ledger from before ``chunking_fix``. The hook warns
    that the checkpoint is behind, never calls the migration, and hands Lightning the
    state dict exactly as saved.
    """
    from anemoi.models.migrations import Migrator

    recorded = [migration.name for migration in Migrator()._grouped_migrations[-1]][:2]
    assert not any("chunking_fix" in name for name in recorded), "fixture assumes chunking_fix is not recorded"

    model = _SmallNet()
    module = _lightning_module_wrapping(model)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in model.state_dict().items()},
        "pytorch-lightning_version": "2.6.5",
        "migrations": _ledger(*recorded),
        "hyper_parameters": {
            # NoOpProcessor: declares neither num_layers nor num_chunks.
            "config": SimpleNamespace(model=SimpleNamespace(processor=SimpleNamespace())),
            "data_indices": {"data": _index_collection({"t2m": 0})},
        },
    }
    saved = {key: value.clone() for key, value in checkpoint["state_dict"].items()}

    with caplog.at_level(logging.WARNING):
        BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    spy_chunking_migration.assert_not_called()
    assert "behind the installed anemoi-models" in caplog.text
    assert "_migration_applied" not in checkpoint
    for key, value in saved.items():
        assert torch.equal(checkpoint["state_dict"][key], value), f"{key} was rewritten"
    assert module._ckpt_model_name_to_index == {"data": {"t2m": 0}}

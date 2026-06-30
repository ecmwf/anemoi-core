# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.tasks.forecaster import Forecaster
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.train.train import _reject_unsupported_warm_start


class DummyIndex:
    def __init__(self) -> None:
        self.name_to_index: dict[str, int] = {}


class DummyIndexWithCompare(DummyIndex):
    """DummyIndex that tracks compare_variables calls."""

    def __init__(self) -> None:
        super().__init__()
        self.compare_called_with: list[tuple] = []

    def compare_variables(self, ckpt_index: dict, data_index: dict) -> None:
        """Track that compare was called."""
        self.compare_called_with.append((ckpt_index, data_index))


class DummyProcessor(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor([value], dtype=torch.float32))

    def forward(self, x, *args, **kwargs) -> torch.Tensor:  # noqa: ANN001
        del args, kwargs
        return x


class DummyModel(torch.nn.Module):
    def __init__(self, lead_times: list[str], offset: float) -> None:
        super().__init__()
        self.pre_processors = torch.nn.ModuleDict({"data": Processors([["dummy", DummyProcessor(offset)]])})
        self.post_processors = torch.nn.ModuleDict(
            {"data": Processors([["dummy", DummyProcessor(offset + 100)]], inverse=True)},
        )

        pre_tend = StepwiseProcessors(lead_times)
        post_tend = StepwiseProcessors(lead_times)
        for idx, lead_time in enumerate(lead_times):
            pre_tend.set(lead_time, Processors([["dummy", DummyProcessor(offset + idx)]]))
            post_tend.set(
                lead_time,
                Processors([["dummy", DummyProcessor(offset + idx + 50)]], inverse=True),
            )

        self.pre_processors_tendencies = torch.nn.ModuleDict({"data": pre_tend})
        self.post_processors_tendencies = torch.nn.ModuleDict({"data": post_tend})


class DummyTrainingModule(BaseTrainingModule):
    def __init__(self) -> None:
        pass

    def _step(self, batch, validation_mode: bool = False) -> Never:  # noqa: ANN001
        raise NotImplementedError


def _make_update_cfg(states: bool, tendencies: bool) -> SimpleNamespace:
    return SimpleNamespace(states=states, tendencies=tendencies)


def _make_dummy_module(model: torch.nn.Module, update_states: bool, update_tendencies: bool) -> DummyTrainingModule:
    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    module.task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module.model = model
    module._device = torch.device("cpu")
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=_make_update_cfg(update_states, update_tendencies)),
    )
    return module


def test_on_load_checkpoint_rebuilds_tendency_processors_for_fewer_steps() -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }

    module = _make_dummy_module(new_model, update_states=False, update_tendencies=True)

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    state_dict = checkpoint["state_dict"]
    assert not any(
        "18h" in key for key in state_dict if key.startswith("model.pre_processors_tendencies.")
    ), "Extra tendency processors from the checkpoint should be dropped."

    new_state = new_model.state_dict()
    old_state = old_model.state_dict()
    for key, value in new_state.items():
        full_key = f"model.{key}"
        if full_key.startswith(("model.pre_processors_tendencies.", "model.post_processors_tendencies.")):
            assert torch.equal(state_dict[full_key], value)
        elif full_key.startswith(("model.pre_processors.", "model.post_processors.")):
            assert torch.equal(state_dict[full_key], old_state[key])


def test_on_load_checkpoint_keeps_checkpoint_processors_when_disabled() -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }

    module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    state_dict = checkpoint["state_dict"]
    assert any(
        "18h" in key for key in state_dict if key.startswith("model.pre_processors_tendencies.")
    ), "Checkpoint tendency processors should be preserved when rebuilding is disabled."

    old_state = old_model.state_dict()
    for key, value in old_state.items():
        full_key = f"model.{key}"
        if full_key.startswith(
            (
                "model.pre_processors.",
                "model.post_processors.",
                "model.pre_processors_tendencies.",
                "model.post_processors_tendencies.",
            ),
        ):
            assert torch.equal(state_dict[full_key], value)


def test_on_load_checkpoint_skips_parity_when_pipeline_already_applied() -> None:
    """The Lightning hook skips parity when the pipeline already applied the checkpoint.

    When ``weights_initialized`` is set, the pipeline already ran weights + parity at
    model-build, so ``on_load_checkpoint`` must not redo them (Lightning's ckpt_path
    restores only optimizer/loop state). Here the processor refresh must be a no-op.
    """
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }
    before = {key: value.clone() for key, value in checkpoint["state_dict"].items()}

    module = _make_dummy_module(new_model, update_states=True, update_tendencies=True)
    module.weights_initialized = True  # the pipeline already applied weights + parity

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    # Parity skipped: the checkpoint state dict is untouched and no metadata restored.
    for key, value in before.items():
        assert torch.equal(checkpoint["state_dict"][key], value)
    assert not hasattr(module, "_ckpt_model_name_to_index")


def test_validate_transfer_learning_add_dataset() -> None:
    """Test adding a new dataset during transfer learning (Scenario A → A+B)."""
    # Setup: checkpoint has ERA5, config has ERA5 + CERRA
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    cerra_index = DummyIndexWithCompare()
    cerra_index.name_to_index = {"t2m": 0, "tp": 1}

    trainer = SimpleNamespace(data_indices={"era5": era5_index, "cerra": cerra_index})
    model = SimpleNamespace(_ckpt_model_name_to_index={"era5": {"t2m": 0, "u10": 1}})

    # Call validation method
    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    # Assert: compare_variables was called for ERA5 (found in checkpoint)
    assert len(era5_index.compare_called_with) == 1
    # Assert: compare_variables was NOT called for CERRA (not in checkpoint)
    assert len(cerra_index.compare_called_with) == 0


def test_validate_transfer_learning_swap_datasets() -> None:
    """Test swapping datasets during transfer learning (Scenario A+B -> A+C)."""
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    icon_index = DummyIndexWithCompare()
    icon_index.name_to_index = {"t2m": 0, "msl": 1}

    trainer = SimpleNamespace(data_indices={"era5": era5_index, "icon": icon_index})
    model = SimpleNamespace(
        _ckpt_model_name_to_index={
            "era5": {"t2m": 0, "u10": 1},
            "cerra": {"t2m": 0, "tp": 1},
        },
    )

    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    assert len(era5_index.compare_called_with) == 1
    assert len(icon_index.compare_called_with) == 0
    assert era5_index.compare_called_with[0] == ({"t2m": 0, "u10": 1}, {"t2m": 0, "u10": 1})


def test_validate_transfer_learning_non_dict_checkpoint_format_returns_early() -> None:
    """Test early return when checkpoint uses non multi-dataset format."""
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    trainer = SimpleNamespace(data_indices={"era5": era5_index})
    model = SimpleNamespace(_ckpt_model_name_to_index={"t2m": 0, "u10": 1})

    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    assert len(era5_index.compare_called_with) == 0


def test_validate_transfer_learning_remove_dataset() -> None:
    """Test removing a dataset during transfer learning (Scenario A+B → A)."""
    # Setup: checkpoint has ERA5 + CERRA, config has only ERA5
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    trainer = SimpleNamespace(data_indices={"era5": era5_index})
    model = SimpleNamespace(
        _ckpt_model_name_to_index={
            "era5": {"t2m": 0, "u10": 1},
            "cerra": {"t2m": 0, "tp": 1},
        },
    )

    # Call validation method
    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    # Assert: compare_variables was called for ERA5
    assert len(era5_index.compare_called_with) == 1
    # Method completes without error (CERRA is silently ignored)


# ── Rollout state persistence across checkpoint save / load ───────────────────


def _make_module_with_forecaster_task(rollout_cfg: dict) -> tuple[DummyTrainingModule, Forecaster]:
    """Build a minimal DummyTrainingModule whose task is a Forecaster."""
    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout=rollout_cfg)
    module.task = task
    module.config = SimpleNamespace(  # type: ignore[assignment]
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=_make_update_cfg(False, False)),
    )
    return module, task


def test_on_save_checkpoint_persists_rollout_step() -> None:
    """on_save_checkpoint writes the current rollout step and last_increased_epoch into the checkpoint."""
    module, task = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    task.on_train_epoch_end(0)
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3

    checkpoint: dict = {}
    BaseTrainingModule.on_save_checkpoint(module, checkpoint)

    assert checkpoint["task_state"]["rollout"]["step"] == 3
    assert checkpoint["task_state"]["rollout"]["last_increased_epoch"] == 1


def test_on_load_checkpoint_restores_rollout_step() -> None:
    """on_load_checkpoint recovers rollout.step so resume continues from the right value."""
    module, task = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})

    checkpoint = {
        "task_state": {"rollout": {"step": 3, "last_increased_epoch": 1}},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
        "state_dict": {},
    }
    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert task.rollout.step == 3
    assert task.rollout._last_increased_epoch == 1


def test_rollout_step_not_spuriously_incremented_on_resume() -> None:
    """PyTorch-Lightning fires on_train_epoch_end with the last completed epoch during restore."""
    rollout_cfg = {"start": 1, "epoch_increment": 1, "maximum": 10}

    # --- first job: two epochs ---
    module, task = _make_module_with_forecaster_task(rollout_cfg)
    task.on_train_epoch_end(0)
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3

    checkpoint: dict = {}
    BaseTrainingModule.on_save_checkpoint(module, checkpoint)

    # --- restore into a fresh module via on_load_checkpoint ---
    resumed_module, resumed_task = _make_module_with_forecaster_task(rollout_cfg)
    checkpoint["hyper_parameters"] = {"data_indices": {"data": DummyIndex()}}
    checkpoint["state_dict"] = {}
    BaseTrainingModule.on_load_checkpoint(resumed_module, checkpoint)

    # PL fires on_train_epoch_end with the last completed epoch during restore
    resumed_task.on_train_epoch_end(1)
    assert resumed_task.rollout.step == 3, "spurious on_train_epoch_end(1) on restore must not increment step"

    # --- second job: two more epochs ---
    resumed_task.on_train_epoch_end(2)
    assert resumed_task.rollout.step == 4
    resumed_task.on_train_epoch_end(3)
    assert resumed_task.rollout.step == 5


def test_on_load_checkpoint_without_task_state_leaves_rollout_at_start() -> None:
    """Checkpoints from before this fix (no task_state key) load without error."""
    module, task = _make_module_with_forecaster_task({"start": 2, "epoch_increment": 1, "maximum": 5})

    checkpoint = {
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
        "state_dict": {},
    }
    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert task.rollout.step == 2


# --- Tests for _validate_transfer_learning_units ---


def test_validate_transfer_learning_units_compatible() -> None:
    """Test that compatible units pass without error."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
            "u10": {"units": "m s**-1"},
        },
    }
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "K"},
                "u10": {"units": "m s**-1"},
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    # Should not raise
    AnemoiTrainer._validate_transfer_learning_units(trainer, model)


def test_validate_transfer_learning_units_incompatible() -> None:
    """Test that incompatible units raise ValueError."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
            "u10": {"units": "m s**-1"},
        },
    }
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "C"},
                "u10": {"units": "m s**-1"},
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    with pytest.raises(ValueError, match="dataset 'era5'"):
        AnemoiTrainer._validate_transfer_learning_units(trainer, model)


def test_validate_transfer_learning_units_missing_checkpoint_metadata() -> None:
    """Test that missing checkpoint variables_metadata produces a warning but no error."""
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "K"},
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=None)

    # Should not raise, just warn
    AnemoiTrainer._validate_transfer_learning_units(trainer, model)


def test_validate_transfer_learning_units_missing_dataset_metadata() -> None:
    """Test that missing dataset variables_metadata produces a warning but no error."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
        },
    }
    datamodule_metadata = {
        "era5": {},  # No variables_metadata
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    # Should not raise, just warn
    AnemoiTrainer._validate_transfer_learning_units(trainer, model)


def test_validate_transfer_learning_units_mismatched_variables_raises() -> None:
    """Test that differing variable sets raise ValueError."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
            "u10": {"units": "m s**-1"},
        },
    }
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "K"},
                "v10": {"units": "m s**-1"},  # Different variable, not in checkpoint
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    # Should raise: variable sets differ (u10 missing, v10 added)
    with pytest.raises(ValueError, match="dataset 'era5'"):
        AnemoiTrainer._validate_transfer_learning_units(trainer, model)


def test_validate_transfer_learning_units_dataset_not_in_checkpoint() -> None:
    """Test that datasets present in config but not in checkpoint are skipped."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
        },
    }
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "K"},
            },
        },
        "cerra": {
            "variables_metadata": {
                "t2m": {"units": "C"},  # Different unit, but dataset not in checkpoint
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    # Should not raise: cerra is not in checkpoint
    AnemoiTrainer._validate_transfer_learning_units(trainer, model)


# --- Tests for the opt-in checkpoint pipeline path (training.checkpoint) ---


def test_checkpoint_pipeline_configured_detects_training_checkpoint() -> None:
    """``_checkpoint_pipeline_configured`` is True only when training.checkpoint is set."""
    configured = SimpleNamespace(
        config=OmegaConf.create({"training": {"checkpoint": {"loading": {"_target_": "x"}}}}),
    )
    assert AnemoiTrainer._checkpoint_pipeline_configured(configured) is True

    absent = SimpleNamespace(config=OmegaConf.create({"training": {}}))
    assert AnemoiTrainer._checkpoint_pipeline_configured(absent) is False


def test_load_via_checkpoint_pipeline_fills_model_weights(tmp_path: Path) -> None:
    """The opt-in pipeline path resolves the run checkpoint and fills the model in place.

    Exercises the trainer-side wiring end to end: the RunSource resolves ``run_id``
    into ``<root.parent>/<run_id>/last.ckpt`` and loads it, and the WeightsOnlyLoader
    fills the existing model's parameter slots (fill-model semantics — same object,
    no re-instantiation).
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    new_state = {key: torch.randn_like(value) for key, value in model.state_dict().items()}

    run_id = "run_A"
    ckpt_dir = tmp_path / run_id
    ckpt_dir.mkdir(parents=True)
    torch.save({"state_dict": new_state}, ckpt_dir / "last.ckpt")

    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {
                        "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
                        "run_id": run_id,
                    },
                    "loading": {
                        "_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader",
                        "strict": False,
                    },
                },
            },
            "system": {
                "output": {"checkpoints": {"root": str(ckpt_dir)}},
            },
        },
    )

    data_indices = {"data": DummyIndex()}
    trainer = SimpleNamespace(
        config=cfg,
        data_indices=data_indices,
        parent_run_server2server=None,
        fork_run_server2server=None,
        _validate_transfer_learning_datasets=lambda _model: None,
        _validate_transfer_learning_units=lambda _model: None,
    )

    result = AnemoiTrainer._load_via_checkpoint_pipeline(trainer, model)

    assert result is model
    for key, value in new_state.items():
        assert torch.equal(result.state_dict()[key], value)
    assert result.data_indices is data_indices
    # The source stage records the resolved path; the trainer caches it for
    # Lightning's ckpt_path resume (consumed by last_checkpoint).
    assert trainer._resolved_ckpt_path == ckpt_dir / "last.ckpt"


def test_load_via_checkpoint_pipeline_keeps_current_data_indices_over_checkpoint(tmp_path: Path) -> None:
    """``data_indices`` after a load is the CURRENT config's, never the checkpoint's.

    ``data_indices`` decides which physical variable each model channel maps to, so a
    silent swap to the checkpoint's mapping would corrupt every downstream forecast
    without any error. This guards the trainer-side assignment
    (``loaded_model.data_indices = self.data_indices``): even when the checkpoint
    carries a *different* ``data_indices``, the live run's indices must win, and the
    checkpoint's are retained only as ``_ckpt_model_name_to_index`` for the
    transfer-learning compatibility validators.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)

    run_id = "run_indices"
    ckpt_dir = tmp_path / run_id
    ckpt_dir.mkdir(parents=True)

    # The checkpoint carries a DIFFERENT variable -> index mapping than the live run.
    checkpoint_index = DummyIndex()
    checkpoint_index.name_to_index = {"t2m": 0, "u10": 1}
    torch.save(
        {
            "state_dict": {key: torch.randn_like(value) for key, value in model.state_dict().items()},
            "hyper_parameters": {"data_indices": {"data": checkpoint_index}},
        },
        ckpt_dir / "last.ckpt",
    )

    # The live run uses a different mapping; it must be the one that survives.
    current_index = DummyIndex()
    current_index.name_to_index = {"z500": 0, "msl": 1}
    current_data_indices = {"data": current_index}

    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {
                        "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
                        "run_id": run_id,
                    },
                    "loading": {
                        "_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader",
                        "strict": False,
                    },
                },
            },
            "system": {"output": {"checkpoints": {"root": str(ckpt_dir)}}},
        },
    )

    trainer = SimpleNamespace(
        config=cfg,
        data_indices=current_data_indices,
        parent_run_server2server=None,
        fork_run_server2server=None,
        _validate_transfer_learning_datasets=lambda _model: None,
        _validate_transfer_learning_units=lambda _model: None,
    )

    result = AnemoiTrainer._load_via_checkpoint_pipeline(trainer, model)

    # The current config's indices win — not the checkpoint's.
    assert result.data_indices is current_data_indices
    assert result.data_indices["data"].name_to_index == {"z500": 0, "msl": 1}
    # The checkpoint's mapping is quarantined for compatibility checks, never promoted
    # to model.data_indices.
    assert result._ckpt_model_name_to_index == {"data": {"t2m": 0, "u10": 1}}


def test_validate_transfer_learning_units_ignore_units_option() -> None:
    """Test that ignore_units=True suppresses an otherwise-failing unit check."""
    ckpt_variables_metadata = {
        "era5": {
            "t2m": {"units": "K"},
            "u10": {"units": "m s**-1"},
        },
    }
    datamodule_metadata = {
        "era5": {
            "variables_metadata": {
                "t2m": {"units": "C"},
                "u10": {"units": "m s**-1"},
            },
        },
    }
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {"check_variables_compatibility": {"ignore_units": True}}}),
        datamodule=SimpleNamespace(metadata=datamodule_metadata),
    )
    model = SimpleNamespace(_ckpt_variables_metadata=ckpt_variables_metadata)

    # Should not raise because ignore_units=True
    AnemoiTrainer._validate_transfer_learning_units(trainer, model)


# --- Keyless neutrality: the default-surface flip must not change keyless runs ---


def test_checkpoint_pipeline_configured_false_when_keyless() -> None:
    """No ``training.checkpoint`` key (or an explicit null) is not pipeline-configured."""
    empty = SimpleNamespace(config=OmegaConf.create({"training": {}}))
    assert AnemoiTrainer._checkpoint_pipeline_configured(empty) is False

    explicit_none = SimpleNamespace(config=OmegaConf.create({"training": {"checkpoint": None}}))
    assert AnemoiTrainer._checkpoint_pipeline_configured(explicit_none) is False


def test_model_property_keyless_returns_plain_model_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``.model`` property returns the freshly instantiated module unchanged when keyless.

    With no ``training.checkpoint`` block the declarative pipeline does not run and no
    warning fires — a keyless run is a plain fresh run.
    """
    import anemoi.training.train.train as train_module

    sentinel = torch.nn.Linear(2, 2)
    monkeypatch.setattr(train_module, "instantiate_with_runtime_kwargs", lambda *_args, **_kwargs: sentinel)

    cfg = OmegaConf.create({"training": {"method": {"_target_": "unused"}}})
    trainer = SimpleNamespace(
        config=cfg,
        task=object(),
        data_indices={"data": DummyIndex()},
        graph_data=object(),
        metadata={},
        datamodule=SimpleNamespace(statistics={}, statistics_tendencies={}),
        supporting_arrays=object(),
    )
    trainer._checkpoint_pipeline_configured = AnemoiTrainer._checkpoint_pipeline_configured.__get__(trainer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = AnemoiTrainer.model.func(trainer)

    # Same object the instantiation returned: the pipeline branch did not run (a
    # SimpleNamespace has no real ``_load_via_checkpoint_pipeline``, so taking it
    # would raise), and ``simplefilter("error")`` proves silence.
    assert result is sentinel


_LOADERS = "anemoi.training.checkpoint.loading.strategies"


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (None, False),  # keyless: Lightning resume keeps ckpt_path
        ({"loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"}}, True),
        ({"loading": {"_target_": f"{_LOADERS}.TransferLearningLoader"}}, True),
        ({"loading": {"_target_": f"{_LOADERS}.ColdStartLoader"}}, True),
        ({"loading": {"_target_": f"{_LOADERS}.WarmStartLoader"}}, False),  # full restore keeps ckpt_path
        ({"modifiers": [{"_target_": "x"}]}, False),  # freeze-only, no loader: resume
    ],
)
def test_skip_lightning_restore_matches_loading_strategy(
    checkpoint: dict | None,
    expected: bool,
) -> None:
    """ckpt_path is suppressed for weights-style pipeline loads, kept for warm start/resume."""
    training = {"checkpoint": checkpoint} if checkpoint is not None else {}
    trainer = SimpleNamespace(config=OmegaConf.create({"training": training}))
    assert AnemoiTrainer._skip_lightning_restore(trainer) is expected


# --- Run-lineage source: training.checkpoint.source -> internal run identity ---

_RUNSOURCE = "anemoi.training.checkpoint.sources.run.RunSource"
_LOCALSOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"


@pytest.mark.parametrize(
    ("source", "expected_run_id", "expected_fork_run_id"),
    [
        # resume -> run_id only (same MLflow run continues)
        ({"_target_": _RUNSOURCE, "run_id": "abc", "fork": False}, "abc", None),
        # fork -> fork_run_id only, run_id None (fresh MLflow id via fork-solo branch)
        ({"_target_": _RUNSOURCE, "run_id": "base999", "fork": True}, None, "base999"),
        # RunSource with no run id -> no-op
        ({"_target_": _RUNSOURCE, "run_id": None, "fork": False}, None, None),
        # explicit path carries no run identity (fresh run loading an explicit ckpt)
        ({"_target_": _LOCALSOURCE, "path": "/scratch/run/last.ckpt"}, None, None),
    ],
)
def test_derive_run_identity_maps_source_to_internal_keys(
    source: dict,
    expected_run_id: str | None,
    expected_fork_run_id: str | None,
) -> None:
    """The RunSource surface lowers to the same internal triplet the legacy keys produced."""
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {"training": {"run_id": None, "fork_run_id": None, "checkpoint": {"source": source}}},
        ),
    )
    AnemoiTrainer._derive_run_identity(trainer)
    assert trainer.config.training.run_id == expected_run_id
    assert trainer.config.training.fork_run_id == expected_fork_run_id


def test_derive_run_identity_noop_without_checkpoint_source() -> None:
    """With no training.checkpoint.source, the legacy keys are left untouched."""
    trainer = SimpleNamespace(config=OmegaConf.create({"training": {"run_id": "keep", "fork_run_id": None}}))
    AnemoiTrainer._derive_run_identity(trainer)
    assert trainer.config.training.run_id == "keep"


def test_last_checkpoint_returns_resolved_context_path(tmp_path: Path) -> None:
    """last_checkpoint returns the path the source stage resolved during model build."""
    ckpt = tmp_path / "last.ckpt"
    trainer = SimpleNamespace(
        start_from_checkpoint=True,
        model=object(),  # model already built; the pipeline cached the resolved path
        _resolved_ckpt_path=ckpt,
    )
    assert AnemoiTrainer.last_checkpoint.func(trainer) == ckpt


def test_last_checkpoint_none_when_not_starting() -> None:
    """No source configured (start_from_checkpoint False) short-circuits to None."""
    trainer = SimpleNamespace(start_from_checkpoint=False)
    assert AnemoiTrainer.last_checkpoint.func(trainer) is None


def test_last_checkpoint_none_when_source_resolved_nothing() -> None:
    """A configured source that did not record a local path (e.g. remote) yields None."""
    trainer = SimpleNamespace(start_from_checkpoint=True, model=object(), _resolved_ckpt_path=None)
    assert AnemoiTrainer.last_checkpoint.func(trainer) is None


# --- Warm-start guard: WarmStartLoader needs a local/run source ---

_REMOTE_SOURCES = [
    "anemoi.training.checkpoint.sources.s3.S3Source",
    "anemoi.training.checkpoint.sources.http.HTTPSource",
]


def _warm_start_cfg(source: dict | None) -> DictConfig:
    """Build a ``training.checkpoint`` config with WarmStartLoader and an optional source."""
    checkpoint: dict = {"loading": {"_target_": f"{_LOADERS}.WarmStartLoader"}}
    if source is not None:
        checkpoint["source"] = source
    return OmegaConf.create({"training": {"checkpoint": checkpoint}})


@pytest.mark.parametrize("source_target", _REMOTE_SOURCES)
def test_warm_start_rejects_remote_source(source_target: str) -> None:
    """Warm start from S3/HTTP would silently drop optimizer/epoch state, so it must raise."""
    with pytest.raises(CheckpointConfigError, match="Warm start"):
        _reject_unsupported_warm_start(_warm_start_cfg({"_target_": source_target}))


def test_warm_start_rejects_missing_source() -> None:
    """Warm start with no source has nothing to resume from and must raise."""
    with pytest.raises(CheckpointConfigError, match="Warm start"):
        _reject_unsupported_warm_start(_warm_start_cfg(None))


@pytest.mark.parametrize("source_target", [_LOCALSOURCE, _RUNSOURCE])
def test_warm_start_allows_local_and_run_sources(source_target: str) -> None:
    """LocalSource and RunSource resolve to a local ckpt_path, so warm start is allowed."""
    _reject_unsupported_warm_start(_warm_start_cfg({"_target_": source_target}))  # must not raise


@pytest.mark.parametrize("source_target", _REMOTE_SOURCES)
def test_non_warm_start_allows_remote_source(source_target: str) -> None:
    """Weights-only loading from a remote source is fine; the guard only gates warm start."""
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {"_target_": source_target},
                    "loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"},
                },
            },
        },
    )
    _reject_unsupported_warm_start(cfg)  # must not raise

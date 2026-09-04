# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import reject_unsupported_warm_start
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.tasks.forecaster import Forecaster
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.train import AnemoiTrainer


class DummyIndex:
    def __init__(self) -> None:
        self.name_to_index: dict[str, int] = {}


class DummyIndexWithCompare(DummyIndex):
    """DummyIndex that tracks compare_variables calls."""

    def __init__(self) -> None:
        super().__init__()
        self.compare_called_with: list[tuple] = []
        self.compare_allow_subset: list[bool] = []

    def compare_variables(self, ckpt_index: dict, data_index: dict, *, allow_subset: bool = False) -> None:
        """Track that compare was called (and with which allow_subset)."""
        self.compare_called_with.append((ckpt_index, data_index))
        self.compare_allow_subset.append(allow_subset)


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


def test_on_load_checkpoint_applies_corrections_regardless_of_weights_initialized() -> None:
    """The Lightning hook corrects the dict it is handed, whatever ``weights_initialized`` says.

    A resume is loaded once, by ``Trainer.fit(ckpt_path=)``, which re-reads the file and
    hands this hook the fresh dict it is about to load. Skipping the corrections on a flag
    set by some earlier load applied them to a copy Lightning discarded, so a resume of a
    tendency model silently reloaded the checkpoint's stale statistics. The flag must not
    matter here: tendency processors come from the live model, metadata is restored.
    """
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }

    module = _make_dummy_module(new_model, update_states=False, update_tendencies=True)
    module.weights_initialized = True

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    state_dict = checkpoint["state_dict"]
    assert not any(
        "18h" in key for key in state_dict if key.startswith("model.pre_processors_tendencies.")
    ), "Stale tendency processors from the checkpoint must be dropped, not preserved by the flag."
    for key, value in new_model.state_dict().items():
        if key.startswith(("pre_processors_tendencies.", "post_processors_tendencies.")):
            assert torch.equal(state_dict[f"model.{key}"], value)
    assert module._ckpt_model_name_to_index == {"data": DummyIndex().name_to_index}


def test_validate_transfer_learning_add_dataset() -> None:
    """Test adding a new dataset during transfer learning (Scenario A → A+B)."""
    # Setup: checkpoint has ERA5, config has ERA5 + CERRA
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    cerra_index = DummyIndexWithCompare()
    cerra_index.name_to_index = {"t2m": 0, "tp": 1}

    trainer = SimpleNamespace(
        data_indices={"era5": era5_index, "cerra": cerra_index},
        config=OmegaConf.create({"training": {}}),
    )
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

    trainer = SimpleNamespace(
        data_indices={"era5": era5_index, "icon": icon_index},
        config=OmegaConf.create({"training": {}}),
    )
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

    trainer = SimpleNamespace(
        data_indices={"era5": era5_index},
        config=OmegaConf.create({"training": {}}),
    )
    model = SimpleNamespace(_ckpt_model_name_to_index={"t2m": 0, "u10": 1})

    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    assert len(era5_index.compare_called_with) == 0


def test_validate_transfer_learning_remove_dataset() -> None:
    """Test removing a dataset during transfer learning (Scenario A+B → A)."""
    # Setup: checkpoint has ERA5 + CERRA, config has only ERA5
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0, "u10": 1}

    trainer = SimpleNamespace(
        data_indices={"era5": era5_index},
        config=OmegaConf.create({"training": {}}),
    )
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


def test_validate_transfer_learning_forwards_allow_variable_subset_flag() -> None:
    """training.allow_variable_subset flows through to compare_variables (issue #838).

    Fine-tuning into a model with FEWER variables must be opt-in: the trainer reads the
    config flag and forwards it so compare_variables tolerates a strict variable subset.
    """
    ckpt_index = {"t2m": 0, "u10": 1, "v10": 2}
    model = SimpleNamespace(_ckpt_model_name_to_index={"era5": ckpt_index})

    # Gate ON: the flag is forwarded as allow_subset=True.
    era5_on = DummyIndexWithCompare()
    era5_on.name_to_index = {"t2m": 0, "u10": 1}
    trainer_on = SimpleNamespace(
        data_indices={"era5": era5_on},
        config=OmegaConf.create({"training": {"allow_variable_subset": True}}),
    )
    AnemoiTrainer._validate_transfer_learning_datasets(trainer_on, model)
    assert era5_on.compare_allow_subset == [True]

    # Default (flag absent): strict, allow_subset=False.
    era5_off = DummyIndexWithCompare()
    era5_off.name_to_index = {"t2m": 0, "u10": 1}
    trainer_off = SimpleNamespace(
        data_indices={"era5": era5_off},
        config=OmegaConf.create({"training": {}}),
    )
    AnemoiTrainer._validate_transfer_learning_datasets(trainer_off, model)
    assert era5_off.compare_allow_subset == [False]


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

    Exercises the trainer-side wiring end to end: the RunIdSource resolves ``run_id``
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
                        "_target_": "anemoi.training.checkpoint.sources.run.RunIdSource",
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
        # A configured source means the trainer is starting from a checkpoint; the
        # MLflow dry-run gate is what clears this, and that is exercised separately.
        start_from_checkpoint=True,
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
                        "_target_": "anemoi.training.checkpoint.sources.run.RunIdSource",
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
        # A configured source means the trainer is starting from a checkpoint; the
        # MLflow dry-run gate is what clears this, and that is exercised separately.
        start_from_checkpoint=True,
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

_RUNSOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOCALSOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"


@pytest.mark.parametrize(
    ("source", "expected_run_id", "expected_fork_run_id"),
    [
        # resume -> run_id only (same MLflow run continues)
        ({"_target_": _RUNSOURCE, "run_id": "abc", "fork": False}, "abc", None),
        # fork -> fork_run_id only, run_id None (fresh MLflow id via fork-solo branch)
        ({"_target_": _RUNSOURCE, "run_id": "base999", "fork": True}, None, "base999"),
        # RunIdSource with no run id -> no-op
        ({"_target_": _RUNSOURCE, "run_id": None, "fork": False}, None, None),
        # explicit path carries no run identity (fresh run loading an explicit ckpt)
        ({"_target_": _LOCALSOURCE, "path": "/scratch/run/last.ckpt"}, None, None),
    ],
)
def test_run_identity_from_config_maps_source(
    source: dict,
    expected_run_id: str | None,
    expected_fork_run_id: str | None,
) -> None:
    """The RunIdSource surface resolves to the (run_id, fork_run_id) run identity."""
    from anemoi.training.checkpoint.sources.run import run_identity_from_config

    config = OmegaConf.create({"training": {"checkpoint": {"source": source}}})
    assert run_identity_from_config(config) == (expected_run_id, expected_fork_run_id)


def test_run_identity_from_config_noop_without_checkpoint_source() -> None:
    """With no training.checkpoint.source, there is no run identity."""
    from anemoi.training.checkpoint.sources.run import run_identity_from_config

    assert run_identity_from_config(OmegaConf.create({"training": {}})) == (None, None)


class _PipelineTrainer(SimpleNamespace):
    """A trainer stub whose ``model`` builds through the real pipeline path.

    ``last_checkpoint`` reads the path the source stage resolved, so it needs the model
    built first; the stub mirrors ``AnemoiTrainer.model`` by running
    ``_load_via_checkpoint_pipeline`` lazily, which also proves ``last_checkpoint``
    triggers the build when it is read first (as ``AnemoiEvaluator`` does).
    """

    def __init__(self, config: DictConfig, base_model: torch.nn.Module) -> None:
        super().__init__(
            config=config,
            start_from_checkpoint=True,
            data_indices={"data": DummyIndex()},
            parent_run_server2server=None,
            fork_run_server2server=None,
            _validate_transfer_learning_datasets=lambda _model: None,
            _validate_transfer_learning_units=lambda _model: None,
        )
        self._base_model = base_model
        self.pipeline_runs = 0

    @cached_property
    def model(self) -> torch.nn.Module:
        self.pipeline_runs += 1
        return AnemoiTrainer._load_via_checkpoint_pipeline(self, self._base_model)


def _spy_torch_load(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    real_load = torch.load

    def _spy(*args: object, **kwargs: object) -> object:
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    return calls


def test_last_checkpoint_is_the_path_the_run_source_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A RunIdSource resume hands Lightning the ``last.ckpt`` the source stage found, unloaded."""
    root = tmp_path / "ckpts" / "abc"
    ckpt = tmp_path / "ckpts" / "abc" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": torch.nn.Linear(2, 2).state_dict()}, ckpt)
    loads = _spy_torch_load(monkeypatch)
    trainer = _PipelineTrainer(
        OmegaConf.create(
            {
                "training": {"checkpoint": {"source": {"_target_": _RUNSOURCE, "run_id": "abc", "fork": False}}},
                "system": {"output": {"checkpoints": {"root": str(root)}}},
            },
        ),
        torch.nn.Linear(2, 2),
    )

    assert AnemoiTrainer.last_checkpoint.func(trainer) == ckpt.resolve()
    assert trainer.pipeline_runs == 1
    assert loads == []


def test_last_checkpoint_expands_and_resolves_a_tilde_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``~/...`` reaches Lightning expanded and canonical, exactly as the source checked it.

    ``Trainer.fit(ckpt_path=PosixPath('~/runs/last.ckpt'))`` cannot find the file; the path
    the source resolved is the one that is handed over.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    ckpt = tmp_path / "runs" / "last.ckpt"
    ckpt.parent.mkdir()
    torch.save({"state_dict": torch.nn.Linear(2, 2).state_dict()}, ckpt)
    trainer = _PipelineTrainer(
        OmegaConf.create(
            {"training": {"checkpoint": {"source": {"_target_": _LOCALSOURCE, "path": "~/runs/last.ckpt"}}}},
        ),
        torch.nn.Linear(2, 2),
    )

    resolved = AnemoiTrainer.last_checkpoint.func(trainer)

    assert resolved == ckpt.resolve()
    assert "~" not in str(resolved)
    assert resolved.exists()


def test_last_checkpoint_none_when_not_starting() -> None:
    """No source configured (start_from_checkpoint False) short-circuits to None."""
    trainer = SimpleNamespace(start_from_checkpoint=False)
    assert AnemoiTrainer.last_checkpoint.func(trainer) is None


def test_resume_from_s3_keeps_the_download_for_lightning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote resume: the download survives the pipeline and is what Lightning reads.

    ``S3Source.resolve`` downloads to a node-local file and keeps it; ``last_checkpoint``
    is that file; nothing is loaded in the pipeline; the trainer deletes the file after
    training through ``_remove_temporary_checkpoints``.
    """
    import sys
    from types import ModuleType

    def fake_download(_url: str, target: str, *_args: object, **_kwargs: object) -> None:
        torch.save({"state_dict": torch.nn.Linear(2, 2).state_dict()}, target)

    fake_s3 = ModuleType("anemoi.utils.remote.s3")
    fake_s3.download_file = fake_download
    monkeypatch.setitem(sys.modules, "anemoi.utils.remote.s3", fake_s3)
    loads = _spy_torch_load(monkeypatch)
    trainer = _PipelineTrainer(
        OmegaConf.create(
            {
                "training": {
                    "checkpoint": {
                        "source": {
                            "_target_": "anemoi.training.checkpoint.sources.s3.S3Source",
                            "url": "s3://b/k.ckpt",
                        },
                        "loading": {"_target_": f"{_LOADERS}.WarmStartLoader"},
                    },
                },
            },
        ),
        torch.nn.Linear(2, 2),
    )

    resolved = AnemoiTrainer.last_checkpoint.func(trainer)
    try:
        assert resolved is not None
        assert resolved.exists(), "the download must survive the pipeline for Trainer.fit(ckpt_path=)"
        assert resolved == trainer._resolved_checkpoint_path
        assert trainer._temporary_checkpoint_files == [resolved]
        assert loads == []
        assert not getattr(trainer.model, "weights_initialized", False)
    finally:
        AnemoiTrainer._remove_temporary_checkpoints(trainer)
    assert not resolved.exists()


# --- Warm-start guard: a resume needs a source; any source will do ---

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


@pytest.mark.parametrize("source_target", [_LOCALSOURCE, _RUNSOURCE, *_REMOTE_SOURCES])
def test_warm_start_accepts_any_source(source_target: str) -> None:
    """Every source resolves to a local file for Lightning; a remote one is downloaded and kept."""
    reject_unsupported_warm_start(_warm_start_cfg({"_target_": source_target}))  # must not raise


def test_warm_start_rejects_missing_source() -> None:
    """Warm start with no source has nothing to resume from and must raise."""
    with pytest.raises(CheckpointConfigError, match=r"no training\.checkpoint\.source"):
        reject_unsupported_warm_start(_warm_start_cfg(None))


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
    reject_unsupported_warm_start(cfg)  # must not raise


# --- Downloads a source kept for Trainer.fit(ckpt_path=) are deleted after training ---


class _KeepingSource(CheckpointSource):
    """A source that keeps a download at ``_KeepingSource.kept`` (set per test via monkeypatch)."""

    kept: Path

    async def resolve(self, context: CheckpointContext) -> Path:
        self._keep_download(context, self.kept)
        context.checkpoint_path = self.kept
        return self.kept

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        await self._load_from_path(context, await self.resolve(context))
        return context


def test_load_via_checkpoint_pipeline_records_temporary_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer keeps the list of downloads the source stage left on disk."""
    kept = tmp_path / "download.ckpt"
    torch.save({"state_dict": {}}, kept)
    monkeypatch.setattr(_KeepingSource, "kept", kept, raising=False)

    model = torch.nn.Linear(2, 2)
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {"_target_": f"{__name__}._KeepingSource"},
                    "loading": {
                        "_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader",
                        "strict": False,
                    },
                },
            },
        },
    )
    trainer = SimpleNamespace(
        config=cfg,
        start_from_checkpoint=True,
        data_indices={"data": DummyIndex()},
        parent_run_server2server=None,
        fork_run_server2server=None,
        _validate_transfer_learning_datasets=lambda _model: None,
        _validate_transfer_learning_units=lambda _model: None,
    )

    AnemoiTrainer._load_via_checkpoint_pipeline(trainer, model)

    assert trainer._temporary_checkpoint_files == [kept]
    assert kept.exists(), "the download must survive the pipeline for Trainer.fit(ckpt_path=)"


def test_remove_temporary_checkpoints_deletes_the_recorded_downloads(tmp_path: Path) -> None:
    """After training the kept downloads are removed; a missing one is not an error."""
    present = tmp_path / "present.ckpt"
    present.write_bytes(b"x")
    gone = tmp_path / "gone.ckpt"

    trainer = SimpleNamespace(_temporary_checkpoint_files=[present, gone])
    AnemoiTrainer._remove_temporary_checkpoints(trainer)

    assert not present.exists()
    assert trainer._temporary_checkpoint_files == []


def test_remove_temporary_checkpoints_without_a_pipeline_run_is_a_noop() -> None:
    """A keyless run never recorded downloads; cleanup must still be safe to call."""
    trainer = SimpleNamespace()
    AnemoiTrainer._remove_temporary_checkpoints(trainer)
    assert trainer._temporary_checkpoint_files == []


def test_keep_download_registers_the_atexit_backstop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A kept download is registered with atexit so a job that never reaches fit() still cleans up."""
    import atexit

    from anemoi.training.checkpoint.sources.base import remove_temporary_file

    registered: list[tuple[object, tuple]] = []
    monkeypatch.setattr(atexit, "register", lambda func, *args: registered.append((func, args)))
    kept = tmp_path / "download.ckpt"
    kept.write_bytes(b"x")
    monkeypatch.setattr(_KeepingSource, "kept", kept, raising=False)

    context = CheckpointContext()
    _KeepingSource()._keep_download(context, kept)

    assert registered == [(remove_temporary_file, (kept,))]
    assert context.temporary_files == [kept]


def _training_config_for_train() -> DictConfig:
    """The config keys ``AnemoiTrainer.train`` reads before and after ``trainer.fit``."""
    return OmegaConf.create(
        {
            "training": {
                "deterministic": False,
                "precision": "32",
                "max_epochs": 1,
                "max_steps": None,
                "num_sanity_val_steps": 0,
                "accum_grad_batches": 1,
                "gradient_clip": {"val": 0.0, "algorithm": "value"},
            },
            "diagnostics": {
                "debug": {"anomaly_detection": False},
                "log": {"interval": 1},
                "enable_progress_bar": False,
                "print_memory_summary": False,
            },
            "system": {"hardware": {"num_gpus_per_node": 1, "num_nodes": 1}},
            "dataloader": {"limit_batches": {"training": 1, "validation": 1}},
            "model": {},
        },
    )


@pytest.mark.parametrize("fit_raises", [False, True])
def test_train_removes_kept_downloads_after_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fit_raises: bool,
) -> None:
    """``train()`` deletes the kept downloads once ``fit`` returns, and also when it raises."""
    import anemoi.training.train.train as train_module

    kept = tmp_path / "download.ckpt"
    kept.write_bytes(b"x")
    fit_calls: list[dict] = []

    class _FakeTrainer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(self, **kwargs: object) -> None:
            fit_calls.append(kwargs)
            if fit_raises:
                msg = "boom"
                raise RuntimeError(msg)

    monkeypatch.setattr(train_module.pl, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_module, "prepare_compilation", lambda model, *_args: model)
    trainer = SimpleNamespace(
        config=_training_config_for_train(),
        accelerator="cpu",
        callbacks=[],
        strategy=None,
        profiler=None,
        logger=False,
        model=torch.nn.Linear(2, 2),
        fit_parameters={"ckpt_path": kept},
        _temporary_checkpoint_files=[kept],
    )
    trainer._remove_temporary_checkpoints = AnemoiTrainer._remove_temporary_checkpoints.__get__(trainer)

    if fit_raises:
        with pytest.raises(RuntimeError, match="boom"):
            AnemoiTrainer.train(trainer)
    else:
        AnemoiTrainer.train(trainer)

    assert fit_calls == [{"ckpt_path": kept}]
    assert not kept.exists()
    assert trainer._temporary_checkpoint_files == []


@pytest.mark.parametrize(
    ("loading", "expect_ckpt_path"),
    [
        (None, True),
        (f"{_LOADERS}.WarmStartLoader", True),
        (f"{_LOADERS}.WeightsOnlyLoader", False),
    ],
)
def test_fit_parameters_hands_the_resolved_path_to_lightning_only_on_resume(
    tmp_path: Path,
    loading: str | None,
    expect_ckpt_path: bool,
) -> None:
    """``Trainer.fit(ckpt_path=)`` receives ``last_checkpoint`` on a resume and ``None`` after a pipeline load."""
    resolved = tmp_path / "last.ckpt"
    checkpoint: dict = {"source": {"_target_": _LOCALSOURCE, "path": str(resolved)}}
    if loading is not None:
        checkpoint["loading"] = {"_target_": loading}
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {"checkpoint": checkpoint}}),
        model=torch.nn.Linear(2, 2),
        datamodule=object(),
        last_checkpoint=resolved,
    )
    trainer._skip_lightning_restore = AnemoiTrainer._skip_lightning_restore.__get__(trainer)

    params = AnemoiTrainer.fit_parameters.func(trainer)

    assert params["ckpt_path"] == (resolved if expect_ckpt_path else None)
    assert params["model"] is trainer.model


def test_on_load_checkpoint_writes_a_replaced_dict_back_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lightning holds the dict it passed in; a replacement from the corrections must land in that object."""
    import anemoi.training.train.methods.base as methods_base

    replacement = {
        "state_dict": {"replaced": torch.ones(1)},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }
    monkeypatch.setattr(methods_base, "apply_checkpoint_corrections", lambda *_args, **_kwargs: replacement)
    module = _make_dummy_module(DummyModel(["6h"], offset=1.0), update_states=False, update_tendencies=False)
    checkpoint = {"state_dict": {"stale": torch.zeros(1)}, "hyper_parameters": {"data_indices": {"data": DummyIndex()}}}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert "stale" not in checkpoint["state_dict"]
    assert torch.equal(checkpoint["state_dict"]["replaced"], torch.ones(1))


# --- The resume hook reports an incomplete migration ledger (finding 3 on the Lightning path) ---


def _shipped_migration_names() -> list[str]:
    from anemoi.models.migrations import Migrator

    return [migration.name for migration in Migrator()._grouped_migrations[-1]]


def _ledger(*names: str) -> list[dict]:
    from anemoi.models.migrations.migrator import MigrationMetadata

    return [
        {
            "name": name,
            "metadata": MigrationMetadata(versions={"migration": "1.0.0", "anemoi-models": "0.11.0"}),
            "signature": f"signature-of-{name}",
        }
        for name in names
    ]


def _resume_checkpoint(names: list[str]) -> dict:
    return {
        "state_dict": {},
        "pytorch-lightning_version": "2.6.5",
        "migrations": _ledger(*names),
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }


def test_on_load_checkpoint_warns_about_an_incomplete_ledger_and_names_the_resume_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On a resume the hook is the only correction pass, so the ledger check lives there too.

    The warning names the outstanding migrations and the file Lightning is resuming
    from (``trainer.ckpt_path``), and the load proceeds: metadata is still restored.
    """
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    module._trainer = SimpleNamespace(ckpt_path="/runs/abc/last.ckpt")
    checkpoint = _resume_checkpoint(_shipped_migration_names()[:3])

    with caplog.at_level(logging.WARNING):
        BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert "behind the installed anemoi-models" in caplog.text
    for name in _shipped_migration_names()[3:]:
        assert name in caplog.text
    assert "anemoi-models migration sync /runs/abc/last.ckpt" in caplog.text
    assert module._ckpt_model_name_to_index == {"data": {}}


def test_on_load_checkpoint_is_quiet_for_an_up_to_date_ledger(caplog: pytest.LogCaptureFixture) -> None:
    """The common case, a checkpoint written by the installed anemoi-models, produces no warning."""
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    checkpoint = _resume_checkpoint(_shipped_migration_names())

    with caplog.at_level(logging.WARNING):
        BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert "behind the installed anemoi-models" not in caplog.text

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
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import reject_unsupported_warm_start
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.runner import checkpoint_pipeline_configured
from anemoi.training.checkpoint.runner import run_checkpoint_pipeline
from anemoi.training.tasks.forecaster import Forecaster
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils import seeding
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed


class DummyIndex:
    def __init__(self) -> None:
        self.name_to_index: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _reset_base_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the base seed that restoring a checkpoint sets from leaking into other tests."""
    monkeypatch.setattr(seeding, "_base_seed", None)


def _run_metadata(base_seed: int = 7) -> dict:
    """Seed entries of the run metadata, as AnemoiTrainer builds them before a checkpoint is restored."""
    seeds = {"base_seed": base_seed, "seed": derive_seed(base_seed, SeedContext.TRAINER)}
    return {**seeds, "metadata_inference": dict(seeds)}


def _resume_hyper_parameters(base_seed: int = 4321) -> dict:
    """Hyperparameters of a checkpoint trained with the given base seed."""
    return {"data_indices": {"data": DummyIndex()}, "metadata": {"base_seed": base_seed}}


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


def _make_update_cfg(states: bool, tendencies: bool, residuals: bool = False) -> SimpleNamespace:
    return SimpleNamespace(states=states, tendencies=tendencies, residuals=residuals)


def _make_dummy_module(model: torch.nn.Module, update_states: bool, update_tendencies: bool) -> DummyTrainingModule:
    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    module.task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module.model = model
    module._device = torch.device("cpu")
    module.config = SimpleNamespace(
        training=SimpleNamespace(
            seed=None,
            update_ds_stats_on_ckpt_load=_make_update_cfg(update_states, update_tendencies),
        ),
    )
    module._set_hparams({"metadata": _run_metadata()})
    return module


def test_on_load_checkpoint_rebuilds_tendency_processors_for_fewer_steps() -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": _resume_hyper_parameters(),
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
        "hyper_parameters": _resume_hyper_parameters(),
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


class DummyResidualModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.pre_processors = torch.nn.ModuleDict({"out": Processors([["dummy", DummyProcessor(offset)]])})
        self.post_processors = torch.nn.ModuleDict(
            {"out": Processors([["dummy", DummyProcessor(offset)]], inverse=True)},
        )
        self.pre_processors_residual = torch.nn.ModuleDict({"out": Processors([["dummy", DummyProcessor(offset)]])})
        self.post_processors_residual = torch.nn.ModuleDict(
            {"out": Processors([["dummy", DummyProcessor(offset)]], inverse=True)},
        )


_RESIDUAL_PREFIXES = ("model.pre_processors_residual.", "model.post_processors_residual.")


@pytest.mark.parametrize(
    ("update_cfg", "expect_refreshed"),
    [
        (SimpleNamespace(states=False, tendencies=False, residuals=True), True),
        (SimpleNamespace(states=False, tendencies=False, residuals=False), False),
        # Configs written before the flag existed follow the schema default.
        (SimpleNamespace(states=False, tendencies=False), True),
    ],
    ids=["enabled", "disabled", "absent"],
)
def test_on_load_checkpoint_refreshes_residual_processors(update_cfg: SimpleNamespace, expect_refreshed: bool) -> None:
    old_model = DummyResidualModel(offset=10.0)
    new_model = DummyResidualModel(offset=1.0)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"out": DummyIndex()}},
    }
    module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)
    module.config.training.update_ds_stats_on_ckpt_load = update_cfg

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    state_dict = checkpoint["state_dict"]
    new_state = new_model.state_dict()
    old_state = old_model.state_dict()
    for key in old_state:
        full_key = f"model.{key}"
        refreshed = expect_refreshed and full_key.startswith(_RESIDUAL_PREFIXES)
        assert torch.equal(state_dict[full_key], new_state[key] if refreshed else old_state[key])


def test_on_load_checkpoint_residual_default_leaves_models_without_residual_processors_alone() -> None:
    """The ``residuals`` default must not pull non-residual models past the early return."""
    old_model = DummyModel(["6h"], offset=10.0)
    new_model = DummyModel(["6h"], offset=1.0)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }
    module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)
    module.config.training.update_ds_stats_on_ckpt_load = SimpleNamespace(states=False, tendencies=False)
    before = {key: value.clone() for key, value in checkpoint["state_dict"].items()}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert checkpoint["state_dict"].keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(checkpoint["state_dict"][key], value)


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


def test_validate_transfer_learning_skips_a_checkpoint_without_dataset_metadata() -> None:
    """A checkpoint that carried no ``data_indices`` leaves nothing to compare, not an AttributeError.

    ``preserve_anemoi_metadata`` sets ``_ckpt_model_name_to_index`` only when the
    checkpoint carries ``hyper_parameters.data_indices``; an inference checkpoint or a
    raw state_dict save (the documented S3 + weights-only recipe) does not. The
    validator runs for every loader, so it has to read the attribute defensively, as
    the units check already does.
    """
    era5_index = DummyIndexWithCompare()
    era5_index.name_to_index = {"t2m": 0}
    trainer = SimpleNamespace(
        data_indices={"era5": era5_index},
        config=OmegaConf.create({"training": {}}),
    )
    model = torch.nn.Linear(2, 2)  # a real module: no attribute, and nn.Module raises AttributeError on access

    AnemoiTrainer._validate_transfer_learning_datasets(trainer, model)

    assert era5_index.compare_called_with == []


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


def _make_module_with_forecaster_task(
    rollout_cfg: dict,
    *,
    seed: int | None = None,
) -> tuple[DummyTrainingModule, Forecaster]:
    """Build a minimal DummyTrainingModule whose task is a Forecaster."""
    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout=rollout_cfg)
    module.task = task
    module.config = SimpleNamespace(  # type: ignore[assignment]
        training=SimpleNamespace(
            seed=seed,
            update_ds_stats_on_ckpt_load=_make_update_cfg(False, False),
        ),
    )
    module._set_hparams({"metadata": _run_metadata()})
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


def test_on_load_checkpoint_overrides_configured_rollout_start(caplog: pytest.LogCaptureFixture) -> None:
    """A full resume uses the checkpoint rollout state instead of rollout.start."""
    module, task = _make_module_with_forecaster_task({"start": 2, "epoch_increment": 2, "maximum": 5})

    checkpoint = {
        "task_state": {"rollout": {"step": 4, "last_increased_epoch": 3}},
        "hyper_parameters": _resume_hyper_parameters(),
        "state_dict": {},
    }
    caplog.set_level(logging.INFO)
    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert task.rollout.step == 4
    assert task.rollout._last_increased_epoch == 3
    assert "Restored rollout step from checkpoint: 4 (task was initialized at step 2)." in caplog.messages


def test_on_train_start_logs_effective_rollout_step(caplog: pytest.LogCaptureFixture) -> None:
    """Training startup reports the final task rollout step."""
    module, _ = _make_module_with_forecaster_task({"start": 2, "epoch_increment": 0, "maximum": 2})
    caplog.set_level(logging.INFO)

    BaseTrainingModule.on_train_start(module)

    assert caplog.messages == ["Effective task rollout step: 2."]


def test_weights_only_pipeline_load_starts_rollout_schedule_from_config() -> None:
    """Loading only weights starts the rollout schedule at rollout.start.

    A weights-only load goes through the checkpoint pipeline at model build and never
    calls ``on_load_checkpoint`` (the trainer withholds ``ckpt_path``), so the task's
    runtime state in the checkpoint is never applied: the rollout schedule starts
    where the config says, not where the checkpoint stopped.
    """
    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

    module, task = _make_module_with_forecaster_task({"start": 2, "epoch_increment": 2, "maximum": 5})
    module.model = torch.nn.Linear(2, 2)
    checkpoint = {
        "task_state": {"rollout": {"step": 4, "last_increased_epoch": 3}},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
        "state_dict": {f"model.{key}": value.clone() for key, value in module.model.state_dict().items()},
    }

    WeightsOnlyLoader(strict=True).process(CheckpointContext(model=module, checkpoint_data=checkpoint))

    assert task.rollout.step == 2
    assert task.rollout._last_increased_epoch == -1

    task.on_train_epoch_end(0)
    assert task.rollout.step == 2
    assert task.rollout._last_increased_epoch == -1

    task.on_train_epoch_end(1)
    assert task.rollout.step == 3
    assert task.rollout._last_increased_epoch == 1


class _RecordingDataModule:
    """Record the time window selected whenever the dataset is refreshed."""

    def __init__(self, task: Forecaster) -> None:
        self._task = task
        self.epoch = 0
        self.offsets = task.get_offsets("training")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sync_dataset_state()

    def sync_dataset_state(self) -> None:
        self.offsets = self._task.get_offsets("training")


def test_on_load_checkpoint_synchronizes_dataloader_time_window() -> None:
    """Datasets are resized for the restored rollout before workers start."""
    module, task = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    datamodule = _RecordingDataModule(task)
    # Simulate a dataset synchronized while the task still has rollout.start.
    datamodule.set_epoch(2)
    assert len(datamodule.offsets) == 2
    module._trainer = SimpleNamespace(datamodule=datamodule)

    checkpoint = {
        "task_state": {"rollout": {"step": 3, "last_increased_epoch": 1}},
        "hyper_parameters": _resume_hyper_parameters(),
        "state_dict": {},
    }
    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert len(datamodule.offsets) == 4
    assert datamodule.epoch == 2


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
    checkpoint["hyper_parameters"] = _resume_hyper_parameters()
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


def test_rollout_schedule_continues_at_configured_interval_after_resume() -> None:
    """A restored rollout still waits for the configured number of completed epochs."""
    rollout_cfg = {"start": 1, "epoch_increment": 2, "maximum": 5}
    module, task = _make_module_with_forecaster_task(rollout_cfg)

    task.on_train_epoch_end(0)
    task.on_train_epoch_end(1)
    assert task.rollout.step == 2

    checkpoint: dict = {}
    BaseTrainingModule.on_save_checkpoint(module, checkpoint)
    checkpoint["hyper_parameters"] = _resume_hyper_parameters()
    checkpoint["state_dict"] = {}

    resumed_module, resumed_task = _make_module_with_forecaster_task(rollout_cfg)
    BaseTrainingModule.on_load_checkpoint(resumed_module, checkpoint)

    resumed_task.on_train_epoch_end(1)
    assert resumed_task.rollout.step == 2
    resumed_task.on_train_epoch_end(2)
    assert resumed_task.rollout.step == 2
    resumed_task.on_train_epoch_end(3)
    assert resumed_task.rollout.step == 3


def test_on_load_checkpoint_without_task_state_leaves_rollout_at_start() -> None:
    """Checkpoints from before this fix (no task_state key) load without error."""
    module, task = _make_module_with_forecaster_task({"start": 2, "epoch_increment": 1, "maximum": 5})

    checkpoint = {
        "hyper_parameters": _resume_hyper_parameters(),
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
    """``checkpoint_pipeline_configured`` is True only when training.checkpoint is set."""
    configured = OmegaConf.create({"training": {"checkpoint": {"loading": {"_target_": "x"}}}})
    assert checkpoint_pipeline_configured(configured) is True

    assert checkpoint_pipeline_configured(OmegaConf.create({"training": {}})) is False


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
    assert checkpoint_pipeline_configured(OmegaConf.create({"training": {}})) is False
    assert checkpoint_pipeline_configured(OmegaConf.create({"training": {"checkpoint": None}})) is False


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
        datamodule=SimpleNamespace(statistics={}, statistics_tendencies={}, residual_statistics={}),
        supporting_arrays=object(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = AnemoiTrainer.model.func(trainer)

    # Same object the instantiation returned: the pipeline branch did not run (a
    # SimpleNamespace has no real ``_load_via_checkpoint_pipeline``, so taking it
    # would raise), and ``simplefilter("error")`` proves silence.
    assert result is sentinel


_LOADERS = "anemoi.training.checkpoint.loading.strategies"


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


@pytest.mark.parametrize("loading_raises", [False, True], ids=["loads", "loader_raises"])
def test_weights_only_from_s3_deletes_the_download_after_loading(
    monkeypatch: pytest.MonkeyPatch,
    loading_raises: bool,
) -> None:
    """A remote weights-only load: the download is loaded, then deleted before the loader runs.

    Nothing is left on disk for the rest of the run, also when the loading stage
    raises, and ``last_checkpoint`` has no path to hand Lightning (a weights-only
    load withholds ``ckpt_path`` anyway).
    """
    import sys
    from types import ModuleType

    from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

    downloads: list[Path] = []

    def fake_download(_url: str, target: str, *_args: object, **_kwargs: object) -> None:
        downloads.append(Path(target))
        torch.save({"state_dict": torch.nn.Linear(2, 2).state_dict()}, target)

    fake_s3 = ModuleType("anemoi.utils.remote.s3")
    fake_s3.download_file = fake_download
    monkeypatch.setitem(sys.modules, "anemoi.utils.remote.s3", fake_s3)

    if loading_raises:

        def _raise(_self: WeightsOnlyLoader, _context: CheckpointContext) -> Never:
            msg = "loader failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(WeightsOnlyLoader, "process", _raise)

    trainer = _PipelineTrainer(
        OmegaConf.create(
            {
                "training": {
                    "checkpoint": {
                        "source": {
                            "_target_": "anemoi.training.checkpoint.sources.s3.S3Source",
                            "url": "s3://b/k.ckpt",
                        },
                        "loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"},
                    },
                },
            },
        ),
        torch.nn.Linear(2, 2),
    )

    try:
        if loading_raises:
            with pytest.raises(Exception, match="loader failed"):
                AnemoiTrainer.last_checkpoint.func(trainer)
        else:
            assert AnemoiTrainer.last_checkpoint.func(trainer) is None
            assert trainer.model.weights_initialized is True

        assert len(downloads) == 1
        assert not downloads[0].exists(), "the download must be deleted straight after loading"
    finally:
        for path in downloads:
            path.unlink(missing_ok=True)


# --- Warm-start guard: a resume needs a source that gives a local file ---

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


@pytest.mark.parametrize("source_target", [_LOCALSOURCE, _RUNSOURCE])
def test_warm_start_accepts_a_local_source(source_target: str) -> None:
    """A source that resolves to a local file can resume a run."""
    reject_unsupported_warm_start(_warm_start_cfg({"_target_": source_target}))  # must not raise


@pytest.mark.parametrize("source_target", _REMOTE_SOURCES)
def test_warm_start_rejects_a_remote_source(source_target: str) -> None:
    """A remote download is deleted after loading, so it cannot resume; the error names LocalSource."""
    with pytest.raises(CheckpointConfigError, match=r"cannot resume a run.*LocalSource"):
        reject_unsupported_warm_start(_warm_start_cfg({"_target_": source_target}))


def test_warm_start_rejects_missing_source() -> None:
    """Warm start with no source has nothing to resume from and must raise."""
    with pytest.raises(CheckpointConfigError, match=r"no training\.checkpoint\.source"):
        reject_unsupported_warm_start(_warm_start_cfg(None))


@pytest.mark.parametrize("source_target", _REMOTE_SOURCES)
def test_non_warm_start_allows_remote_source(source_target: str) -> None:
    """Weights-only loading from a remote source is fine; the guard only gates a resume."""
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
    """``Trainer.fit(ckpt_path=)`` receives the resolved file on a resume and ``None`` after a pipeline load."""
    model = torch.nn.Linear(2, 2)
    resolved = tmp_path / "last.ckpt"
    torch.save({"state_dict": torch.nn.Linear(2, 2).state_dict()}, resolved)
    checkpoint: dict = {"source": {"_target_": _LOCALSOURCE, "path": str(resolved)}}
    if loading is not None:
        checkpoint["loading"] = {"_target_": loading}
    trainer = SimpleNamespace(
        config=OmegaConf.create({"training": {"checkpoint": checkpoint}}),
        start_from_checkpoint=True,
        data_indices=None,
        datamodule=object(),
        _validate_transfer_learning_datasets=lambda _model: None,
        _validate_transfer_learning_units=lambda _model: None,
    )
    trainer.model = AnemoiTrainer._load_via_checkpoint_pipeline(trainer, model)
    trainer.last_checkpoint = AnemoiTrainer.last_checkpoint.func(trainer)

    params = AnemoiTrainer.fit_parameters.func(trainer)

    assert params["ckpt_path"] == (resolved.resolve() if expect_ckpt_path else None)
    assert params["model"] is model


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


def test_on_load_checkpoint_honours_a_migration_that_returns_a_new_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that builds a new checkpoint is honoured through the real hook.

    The hook runs the real corrections; only the edge-permutation migration is
    faked, returning a new object. Lightning loads the dict it passed to the hook,
    so the migrated content must end up in that same object.
    """
    import anemoi.training.checkpoint.loading.base as loading_base

    def rebuilding_migrate(ckpt: dict, _model: torch.nn.Module) -> dict:
        return {**ckpt, "state_dict": {"migrated": torch.ones(1)}}

    monkeypatch.setattr(loading_base, "_trainable_edge_perm_fix", rebuilding_migrate)
    module = _make_dummy_module(DummyModel(["6h"], offset=1.0), update_states=False, update_tendencies=False)
    checkpoint = {"state_dict": {"stale": torch.zeros(1)}, "hyper_parameters": {"data_indices": {"data": DummyIndex()}}}
    held_by_lightning = checkpoint

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert held_by_lightning["state_dict"].keys() == {"migrated"}
    assert torch.equal(held_by_lightning["state_dict"]["migrated"], torch.ones(1))


# --- a resume is one Lightning load, corrected by the hook ---


def _index_with(name_to_index: dict[str, int]) -> DummyIndex:
    index = DummyIndex()
    index.name_to_index = name_to_index
    return index


def _legacy_on_load_checkpoint(module: DummyTrainingModule, checkpoint: dict) -> None:
    """The steps ``BaseTrainingModule.on_load_checkpoint`` ran before the checkpoint pipeline existed.

    Edge-perm migration, then the processor refresh, then ``_ckpt_model_name_to_index``,
    then ``_ckpt_variables_metadata``, without the task-state and datamodule steps that
    need a trainer. The edge-perm migration is a no-op for a model without a
    ``StaticGraphProvider``, so this pins the processor refresh and the metadata.
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
    result must match the legacy hook steps bit for bit.
    """
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": _index_with({"t2m": 0})}},
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

    module = _make_dummy_module(DummyModel(["6h", "12h"], offset=1.0), update_states=False, update_tendencies=True)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {
                        "_target_": "anemoi.training.checkpoint.sources.local.LocalSource",
                        "path": str(ckpt_path),
                    },
                    "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
                },
                "update_ds_stats_on_ckpt_load": {"states": False, "tendencies": True},
            },
        },
    )

    executed = run_checkpoint_pipeline(module, cfg)

    # No pipeline load: the file was only resolved, the model is untouched.
    assert loads == []
    assert executed.checkpoint_path == ckpt_path.resolve()
    assert executed.weights_loaded is False
    for key, value in before.items():
        assert torch.equal(module.state_dict()[key], value)

    # Lightning's ckpt_path load: read the file, run the hook on that dict, load it.
    lightning_dict = real_load(executed.checkpoint_path, weights_only=False, map_location="cpu")
    BaseTrainingModule.on_load_checkpoint(module, lightning_dict)
    module.load_state_dict(lightning_dict["state_dict"], strict=True)

    # The legacy steps, on an identical module and a fresh read of the same file.
    reference = _make_dummy_module(DummyModel(["6h", "12h"], offset=1.0), update_states=False, update_tendencies=True)
    legacy_dict = real_load(ckpt_path, weights_only=False, map_location="cpu")
    _legacy_on_load_checkpoint(reference, legacy_dict)
    reference.load_state_dict(legacy_dict["state_dict"], strict=True)

    resumed = module.state_dict()
    for key, value in reference.state_dict().items():
        assert torch.equal(resumed[key], value), f"{key} differs from the legacy hook path"
    # The tendency processors are the live model's, not the checkpoint's stale 10.x.
    assert torch.equal(
        resumed["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
        torch.tensor([1.0]),
    )
    assert module._ckpt_model_name_to_index == {"data": {"t2m": 0}}
    assert module._ckpt_variables_metadata == reference._ckpt_variables_metadata


@pytest.fixture
def spy_chunking_migration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Observe ``chunking_fix`` where ``loading.base`` calls it."""
    spy = MagicMock(side_effect=lambda ckpt: {**ckpt, "_migration_applied": True})
    monkeypatch.setattr("anemoi.training.checkpoint.loading.base._chunking_fix", spy)
    return spy


def test_resume_hook_does_not_migrate_an_autoencoder_checkpoint(spy_chunking_migration: MagicMock) -> None:
    """On a resume the hook is the only correction pass, so it must skip autoencoder weights too.

    Same checkpoint shape as the pipeline-path test in ``test_format_migrations.py``: a
    ``NoOpProcessor`` config. The hook never calls the migration and hands Lightning
    the state dict exactly as saved.
    """
    module = _make_dummy_module(DummyModel(["6h"], offset=1.0), update_states=False, update_tendencies=False)
    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in module.model.state_dict().items()},
        "pytorch-lightning_version": "2.6.5",
        "hyper_parameters": {
            # NoOpProcessor: declares neither num_layers nor num_chunks.
            "config": SimpleNamespace(model=SimpleNamespace(processor=SimpleNamespace())),
            "data_indices": {"data": _index_with({"t2m": 0})},
        },
    }
    saved = {key: value.clone() for key, value in checkpoint["state_dict"].items()}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    spy_chunking_migration.assert_not_called()
    assert "_migration_applied" not in checkpoint
    for key, value in saved.items():
        assert torch.equal(checkpoint["state_dict"][key], value), f"{key} was rewritten"
    assert module._ckpt_model_name_to_index == {"data": {"t2m": 0}}


def _attach_trainer(module: DummyTrainingModule) -> None:
    """Attach a stand-in for the Trainer that restores the checkpoint in a training run."""
    module._trainer = SimpleNamespace(datamodule=None)


def test_on_load_checkpoint_continues_with_checkpoint_base_seed() -> None:
    """Without a seed in the config, a resumed run trains with the checkpoint's seed and records it."""
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    _attach_trainer(module)
    checkpoint = {"hyper_parameters": _resume_hyper_parameters(base_seed=4321), "state_dict": {}}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert get_base_seed() == 4321
    assert module.hparams["metadata"] == _run_metadata(base_seed=4321)


def test_on_load_checkpoint_keeps_configured_base_seed() -> None:
    """A seed set in the config replaces the checkpoint's seed."""
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5}, seed=1234)
    _attach_trainer(module)
    checkpoint = {"hyper_parameters": _resume_hyper_parameters(base_seed=4321), "state_dict": {}}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert get_base_seed() == 1234
    assert module.hparams["metadata"] == _run_metadata(base_seed=1234)


def test_weights_only_pipeline_load_keeps_run_base_seed() -> None:
    """Loading only the weights starts a new run, which keeps its own seed.

    A weights-only load goes through the checkpoint pipeline and does not call
    ``on_load_checkpoint``, so the checkpoint's base seed is not applied.
    """
    from anemoi.training.checkpoint.loading.strategies import WeightsOnlyLoader

    seeding.set_base_seed(7)
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    _attach_trainer(module)
    module.model = torch.nn.Linear(2, 2)
    checkpoint = {
        "hyper_parameters": _resume_hyper_parameters(base_seed=4321),
        "state_dict": {f"model.{key}": value.clone() for key, value in module.model.state_dict().items()},
    }

    WeightsOnlyLoader(strict=True).process(CheckpointContext(model=module, checkpoint_data=checkpoint))

    assert get_base_seed() == 7
    assert module.hparams["metadata"] == _run_metadata(base_seed=7)


def test_on_load_checkpoint_without_stored_base_seed_keeps_run_base_seed(caplog: pytest.LogCaptureFixture) -> None:
    """A checkpoint from an older version without a stored base seed resumes with the run's own seed."""
    seeding.set_base_seed(7)
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    _attach_trainer(module)
    hyper_parameters = _resume_hyper_parameters()
    del hyper_parameters["metadata"]["base_seed"]
    checkpoint = {"hyper_parameters": hyper_parameters, "state_dict": {}}

    with caplog.at_level(logging.WARNING):
        BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert get_base_seed() == 7
    assert module.hparams["metadata"] == _run_metadata(base_seed=7)
    assert "stores no base seed" in caplog.text


def test_on_load_checkpoint_without_trainer_leaves_base_seed() -> None:
    """Converting a checkpoint loads it without a Trainer and leaves the seeds alone.

    The config stored in a checkpoint trained before ``training.seed`` existed has no seed entry.
    """
    module, _ = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    del module.config.training.seed
    checkpoint = {"hyper_parameters": _resume_hyper_parameters(base_seed=4321), "state_dict": {}}

    BaseTrainingModule.on_load_checkpoint(module, checkpoint)

    assert seeding._base_seed is None
    assert module.hparams["metadata"] == _run_metadata()


@pytest.mark.parametrize(
    ("batches_run", "epoch_advances"),
    [
        pytest.param(6, True, id="all-batches"),
        pytest.param(4, False, id="stopped-early"),
    ],
)
def test_on_train_epoch_end_advances_epoch_state_only_after_full_epoch(
    batches_run: int,
    epoch_advances: bool,
) -> None:
    """Stopping early keeps the rollout and data epoch, so a run resumed inside the epoch uses them."""
    module, task = _make_module_with_forecaster_task({"start": 1, "epoch_increment": 1, "maximum": 5})
    datamodule = _RecordingDataModule(task)
    batch_progress = SimpleNamespace(current=SimpleNamespace(ready=batches_run))
    module._fabric = None
    module._jit_is_scripting = False
    module._trainer = SimpleNamespace(
        datamodule=datamodule,
        current_epoch=0,
        num_training_batches=6,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(batch_progress=batch_progress)),
    )

    BaseTrainingModule.on_train_epoch_end(module)

    assert task.rollout.step == (2 if epoch_advances else 1)
    assert datamodule.epoch == (1 if epoch_advances else 0)

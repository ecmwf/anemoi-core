from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import StaticGraphProvider
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.checkpoint import transfer_learning_loading


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


def _make_static_graph_provider(trainable_size: int = 2) -> StaticGraphProvider:
    graph = HeteroData()
    graph.edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 0]], dtype=torch.long)
    graph.edge_attr = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

    return StaticGraphProvider(
        graph=graph,
        edge_attributes=["edge_attr"],
        src_size=3,
        dst_size=2,
        trainable_size=trainable_size,
    )


class DummyGraphModel(torch.nn.Module):
    def __init__(self, trainable_size: int = 2) -> None:
        super().__init__()
        self.graph_provider = _make_static_graph_provider(trainable_size)


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
    module.model = model
    module._device = torch.device("cpu")
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=_make_update_cfg(update_states, update_tendencies)),
    )
    return module


def _make_minimal_ckpt_config() -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(processor=SimpleNamespace(num_layers=1, num_chunks=1)))


def test_on_load_checkpoint_rebuilds_tendency_processors_for_fewer_steps() -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }

    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    module.model = new_model
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=_make_update_cfg(False, True)),
    )

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

    module = DummyTrainingModule.__new__(DummyTrainingModule)
    torch.nn.Module.__init__(module)
    module.model = new_model
    module.config = SimpleNamespace(
        training=SimpleNamespace(update_ds_stats_on_ckpt_load=_make_update_cfg(False, False)),
    )

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


def test_transfer_learning_loading_updates_processors_when_enabled(
    tmp_path: Path,
) -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    old_module = _make_dummy_module(old_model, update_states=False, update_tendencies=False)
    new_module = _make_dummy_module(new_model, update_states=True, update_tendencies=True)
    new_state_before = new_module.state_dict()

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": {"data": SimpleNamespace(name_to_index={})},
        },
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    transfer_learning_loading(new_module, ckpt_path)

    state_dict = new_module.state_dict()
    assert torch.equal(
        state_dict["model.pre_processors.data.processors.dummy.value"],
        new_state_before["model.pre_processors.data.processors.dummy.value"],
    )
    assert torch.equal(
        state_dict["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
        new_state_before["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
    )


def test_transfer_learning_loading_preserves_checkpoint_processors_when_disabled(
    tmp_path: Path,
) -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    old_module = _make_dummy_module(old_model, update_states=False, update_tendencies=False)
    new_module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": {"data": SimpleNamespace(name_to_index={})},
        },
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    transfer_learning_loading(new_module, ckpt_path)

    state_dict = new_module.state_dict()
    assert torch.equal(
        state_dict["model.pre_processors.data.processors.dummy.value"],
        old_module.state_dict()["model.pre_processors.data.processors.dummy.value"],
    )
    assert torch.equal(
        state_dict["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
        old_module.state_dict()["model.pre_processors_tendencies.data._processors.6h.processors.dummy.value"],
    )


def test_transfer_learning_loading_populates_ckpt_indices_from_dict(tmp_path: Path) -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    old_module = _make_dummy_module(old_model, update_states=False, update_tendencies=False)
    new_module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": {
                "era5": SimpleNamespace(name_to_index={"t2m": 0, "u10": 1}),
                "cerra": SimpleNamespace(name_to_index={"t2m": 0, "tp": 1}),
            },
        },
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    transfer_learning_loading(new_module, ckpt_path)

    assert new_module._ckpt_model_name_to_index == {
        "era5": {"t2m": 0, "u10": 1},
        "cerra": {"t2m": 0, "tp": 1},
    }


def test_transfer_learning_loading_filters_trainable_edge_mismatch_before_migration(tmp_path: Path) -> None:
    new_module = _make_dummy_module(DummyGraphModel(trainable_size=2), update_states=False, update_tendencies=False)
    trainable_key = "model.graph_provider.trainable.trainable"
    layout_version_key = "model.graph_provider.trainable_layout_version"
    trainable_before = new_module.state_dict()[trainable_key].clone()

    state_dict = new_module.state_dict()
    state_dict[trainable_key] = torch.ones(2, 2)
    del state_dict[layout_version_key]

    checkpoint = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": {"data": SimpleNamespace(name_to_index={})},
        },
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    transfer_learning_loading(new_module, ckpt_path)

    assert torch.equal(new_module.state_dict()[trainable_key], trainable_before)
    assert new_module.state_dict()[layout_version_key].item() == 1


def test_transfer_learning_loading_raises_on_old_checkpoint_data_indices_format(tmp_path: Path) -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    old_module = _make_dummy_module(old_model, update_states=False, update_tendencies=False)
    new_module = _make_dummy_module(new_model, update_states=False, update_tendencies=False)

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": SimpleNamespace(name_to_index={"t2m": 0, "u10": 1}),
        },
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(TypeError, match="older version of anemoi-core"):
        transfer_learning_loading(new_module, ckpt_path)


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

    absent = SimpleNamespace(config=OmegaConf.create({"training": {"run_id": "abc"}}))
    assert AnemoiTrainer._checkpoint_pipeline_configured(absent) is False


def test_load_via_checkpoint_pipeline_fills_model_weights(tmp_path: Path) -> None:
    """The opt-in pipeline path resolves the lineage checkpoint and fills the model in place.

    Exercises the trainer-side wiring end to end: the run-lineage resolver turns
    ``run_id`` into ``<root.parent>/<run_id>/last.ckpt``, the LocalSource loads it,
    and the WeightsOnlyLoader fills the existing model's parameter slots
    (fill-model semantics — same object, no re-instantiation).
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
                "run_id": run_id,
                "fork_run_id": None,
                "checkpoint": {
                    "source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource"},
                    "loading": {
                        "_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader",
                        "strict": False,
                    },
                },
            },
            "system": {
                "input": {"warm_start": None},
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


# --- Tests for legacy-key delegation + deprecation (K5) ---


def test_legacy_weights_only_delegates_with_deprecation_warning() -> None:
    """load_weights_only (no TL) translates to a WeightsOnlyLoader block + one warning."""
    trainer = SimpleNamespace(
        load_weights_only=True,
        config=OmegaConf.create({"training": {"transfer_learning": False}}),
    )
    with pytest.warns(FutureWarning, match="WeightsOnlyLoader") as record:
        block = AnemoiTrainer._legacy_checkpoint_config(trainer)

    assert len(record) == 1
    assert block["source"]["_target_"].endswith("LocalSource")
    assert block["loading"]["_target_"].endswith("WeightsOnlyLoader")
    assert "modifiers" not in block


def test_legacy_transfer_learning_delegates_with_deprecation_warning() -> None:
    """transfer_learning translates to a TransferLearningLoader block + one warning."""
    trainer = SimpleNamespace(
        load_weights_only=True,
        config=OmegaConf.create({"training": {"transfer_learning": True}}),
    )
    with pytest.warns(FutureWarning, match="TransferLearningLoader"):
        block = AnemoiTrainer._legacy_checkpoint_config(trainer)

    assert block["loading"]["_target_"].endswith("TransferLearningLoader")
    assert block["loading"]["skip_mismatched"] is True


def test_legacy_submodules_to_freeze_prefixes_model_model_for_rooting_parity() -> None:
    """Freeze names are prefixed ``model.model.`` for rooting parity.

    The modifier (Task-rooted ``get_submodule``) must target the same submodules
    the legacy freeze (rooted at ``model.model.model``) did.
    """
    trainer = SimpleNamespace(
        load_weights_only=False,
        config=OmegaConf.create({"training": {"submodules_to_freeze": ["encoder", "processor.0"]}}),
    )
    with pytest.warns(FutureWarning, match="FreezingModifierStage"):
        block = AnemoiTrainer._legacy_checkpoint_config(trainer)

    modifier = block["modifiers"][0]
    assert modifier["_target_"].endswith("FreezingModifierStage")
    assert modifier["submodules_to_freeze"] == ["model.model.encoder", "model.model.processor.0"]
    # Freeze-only: no acquisition / loading stage.
    assert "source" not in block
    assert "loading" not in block


def test_legacy_checkpoint_config_none_when_no_legacy_key() -> None:
    """No deprecated key set => no delegation block, no warning."""
    trainer = SimpleNamespace(
        load_weights_only=False,
        config=OmegaConf.create({"training": {"transfer_learning": False}}),
    )
    assert AnemoiTrainer._legacy_checkpoint_config(trainer) is None


def test_legacy_weights_only_delegation_loads_via_pipeline(tmp_path: Path) -> None:
    """End to end: load_weights_only delegates through the pipeline and fills the model."""
    torch.manual_seed(1)
    model = torch.nn.Linear(4, 2)
    new_state = {key: torch.randn_like(value) for key, value in model.state_dict().items()}

    run_id = "run_L"
    ckpt_dir = tmp_path / run_id
    ckpt_dir.mkdir(parents=True)
    torch.save({"state_dict": new_state}, ckpt_dir / "last.ckpt")

    cfg = OmegaConf.create(
        {
            "training": {"run_id": run_id, "fork_run_id": None, "transfer_learning": False},
            "system": {"input": {"warm_start": None}, "output": {"checkpoints": {"root": str(ckpt_dir)}}},
        },
    )
    trainer = SimpleNamespace(
        load_weights_only=True,
        config=cfg,
        data_indices={"data": DummyIndex()},
        parent_run_server2server=None,
        fork_run_server2server=None,
        _validate_transfer_learning_datasets=lambda _model: None,
        _validate_transfer_learning_units=lambda _model: None,
    )

    with pytest.warns(FutureWarning, match="WeightsOnlyLoader"):
        block = AnemoiTrainer._legacy_checkpoint_config(trainer)
    result = AnemoiTrainer._load_via_checkpoint_pipeline(trainer, model, checkpoint_config=block)

    assert result is model
    for key, value in new_state.items():
        assert torch.equal(result.state_dict()[key], value)


class _FreezeInner(torch.nn.Module):
    """The innermost nn model (``Task.model.model``) with named submodules."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.processor = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.decoder = torch.nn.Linear(4, 2)


class _FreezeInterface(torch.nn.Module):
    """The model interface (``Task.model``)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FreezeInner()


class _FreezeTask(torch.nn.Module):
    """A Task-like module nesting ``.model.model`` like the real training module."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FreezeInterface()


def test_legacy_freeze_delegation_matches_legacy_on_nested_model() -> None:
    """End-to-end freeze-rooting parity on a real ``Task.model.model`` nesting.

    The delegated ``FreezingModifierStage`` (names prefixed ``model.model.``) must
    freeze the exact same parameters as the legacy ``freeze_submodule_by_name``
    rooted at ``model.model.model``.
    """
    from anemoi.training.utils.checkpoint import freeze_submodule_by_name

    submodules = ["encoder", "processor.0"]

    # Legacy path: freeze rooted at model.model.model.
    legacy_model = _FreezeTask()
    for name in submodules:
        assert freeze_submodule_by_name(legacy_model.model.model, name)

    # Delegated path: legacy keys translated to a FreezingModifierStage and run
    # through the pipeline (freeze-only: no source, no loading stage).
    delegated_model = _FreezeTask()
    trainer = SimpleNamespace(
        load_weights_only=False,
        config=OmegaConf.create({"training": {"submodules_to_freeze": submodules}}),
    )
    with pytest.warns(FutureWarning, match="FreezingModifierStage"):
        block = AnemoiTrainer._legacy_checkpoint_config(trainer)
    AnemoiTrainer._load_via_checkpoint_pipeline(trainer, delegated_model, checkpoint_config=block)

    legacy_grads = {name: param.requires_grad for name, param in legacy_model.named_parameters()}
    delegated_grads = {name: param.requires_grad for name, param in delegated_model.named_parameters()}

    # Identical requires_grad map: the delegation froze exactly what legacy froze.
    assert delegated_grads == legacy_grads
    # Sanity: the targeted submodules are frozen, the others remain trainable.
    assert not delegated_grads["model.model.encoder.weight"]
    assert not delegated_grads["model.model.processor.0.weight"]
    assert delegated_grads["model.model.decoder.weight"]

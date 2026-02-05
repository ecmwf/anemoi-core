from pathlib import Path
from types import SimpleNamespace
from typing import Never

import torch

from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.utils.checkpoint import transfer_learning_loading


class DummyIndex:
    def __init__(self) -> None:
        self.name_to_index: dict[str, int] = {}


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


class DummyGraphModule(BaseGraphModule):
    task_type = "forecaster"

    def __init__(self) -> None:
        pass

    def _step(self, batch, validation_mode: bool = False) -> Never:  # noqa: ANN001
        raise NotImplementedError


def _make_dummy_module(model: torch.nn.Module, update_stats: bool) -> DummyGraphModule:
    module = DummyGraphModule.__new__(DummyGraphModule)
    torch.nn.Module.__init__(module)
    module.model = model
    module._device = torch.device("cpu")
    module.config = SimpleNamespace(training=SimpleNamespace(update_ds_stats_on_ckpt_load=update_stats))
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

    module = DummyGraphModule.__new__(DummyGraphModule)
    torch.nn.Module.__init__(module)
    module.model = new_model
    module.config = SimpleNamespace(training=SimpleNamespace(update_ds_stats_on_ckpt_load=True))

    BaseGraphModule.on_load_checkpoint(module, checkpoint)

    state_dict = checkpoint["state_dict"]
    assert not any(
        "18h" in key for key in state_dict if key.startswith("model.pre_processors_tendencies.")
    ), "Extra tendency processors from the checkpoint should be dropped."

    new_state = new_model.state_dict()
    for key, value in new_state.items():
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


def test_on_load_checkpoint_keeps_checkpoint_processors_when_disabled() -> None:
    old_model = DummyModel(["6h", "12h", "18h"], offset=10.0)
    new_model = DummyModel(["6h", "12h"], offset=1.0)

    checkpoint = {
        "state_dict": {f"model.{key}": value.clone() for key, value in old_model.state_dict().items()},
        "hyper_parameters": {"data_indices": {"data": DummyIndex()}},
    }

    module = DummyGraphModule.__new__(DummyGraphModule)
    torch.nn.Module.__init__(module)
    module.model = new_model
    module.config = SimpleNamespace(training=SimpleNamespace(update_ds_stats_on_ckpt_load=False))

    BaseGraphModule.on_load_checkpoint(module, checkpoint)

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

    old_module = _make_dummy_module(old_model, update_stats=False)
    new_module = _make_dummy_module(new_model, update_stats=True)
    new_state_before = new_module.state_dict()

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": SimpleNamespace(name_to_index={}),
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

    old_module = _make_dummy_module(old_model, update_stats=False)
    new_module = _make_dummy_module(new_model, update_stats=False)

    checkpoint = {
        "state_dict": old_module.state_dict(),
        "hyper_parameters": {
            "config": _make_minimal_ckpt_config(),
            "data_indices": SimpleNamespace(name_to_index={}),
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

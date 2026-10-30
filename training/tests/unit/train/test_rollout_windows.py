# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rollout windows (training.rollout_windows): the rollout cut into windows, each backpropagated on its own."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import Forecaster
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.utils.masks import NoOutputMask

B, E, G, V = 2, 3, 5, 2

Wired = Callable[[int, tuple[int, tuple[float, ...]] | None], EnsembleTraining]


class StepModel(torch.nn.Module):
    """One forecast step: a learnt map of the state's variables. Records the rollout step it is called with."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(0.5 * torch.randn(V, V, generator=generator))
        self.bias = torch.nn.Parameter(0.1 * torch.randn(V, generator=generator))
        self.fcsteps: list[int] = []

    def forward(self, x: dict[str, torch.Tensor], fcstep: int = 0, **_kwargs: Any) -> dict[str, torch.Tensor]:
        self.fcsteps.append(fcstep)
        state = x["data"][:, -1:]
        return {"data": torch.tanh(state @ self.weight + self.bias) + 0.1 * fcstep}


def _module(rollout: int, windows: tuple[int, tuple[float, ...]] | None) -> EnsembleTraining:
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": rollout, "maximum": rollout},
        data_frequency="6h",
    )
    module = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(module)
    module.task = task
    module.config = DictConfig({})
    module.data_indices = {"data": IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"A": 0, "B": 1})}
    module.dataset_names = ["data"]
    module.target_dataset_names = ["data"]
    module.model_comm_group = None
    module.model_comm_group_size = 1
    module.grid_shard_sizes = {"data": None}
    module.grid_shard_slice = {"data": None}
    module.output_mask = {"data": NoOutputMask()}
    module.nens_per_device = E
    module.model = StepModel()
    module.loss = {"data": torch.nn.Module()}  # only its name is read (the per-dataset loss metric)
    module.rollout_windows = windows
    return module


def _batch(rollout: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {"data": torch.randn(B, 1 + rollout, 1, G, V, generator=generator)}


def _target(batch: dict[str, torch.Tensor], rollout_step: int) -> dict[str, torch.Tensor]:
    return {"data": batch["data"][:, 1 + rollout_step : 2 + rollout_step]}


def _loss(y_pred: dict[str, torch.Tensor], y: dict[str, torch.Tensor], **_kwargs: Any) -> tuple:
    return ((y_pred["data"] - y["data"]) ** 2).mean(), {}, y_pred


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Wired:
    """A module whose task reads targets from the batch and whose state becomes each step's prediction."""

    def make(rollout: int, windows: tuple[int, tuple[float, ...]] | None) -> EnsembleTraining:
        module = _module(rollout, windows)
        monkeypatch.setattr(module, "compute_loss_metrics", _loss)
        monkeypatch.setattr(
            module.task,
            "get_targets",
            lambda batch, rollout_step=0, **_kw: _target(batch, rollout_step),
        )
        monkeypatch.setattr(module.task, "advance_input", lambda _x, y_pred, *_a, **_kw: dict(y_pred))
        return module

    return make


def _grads(module: EnsembleTraining) -> list[torch.Tensor]:
    return [p.grad.clone() for p in module.parameters()]


def _windowed(
    module: EnsembleTraining,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list]:
    calls = []

    def backward(loss: torch.Tensor, last: bool) -> None:
        calls.append(last)
        loss.backward()

    total = module._windowed_rollout_backward(batch, backward)
    return total, _grads(module), calls


def _reference(
    module: EnsembleTraining,
    batch: dict[str, torch.Tensor],
    window: int,
    step_weights: list[float],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The weighted loss with the state detached at every window boundary, in one backward pass."""
    x = {"data": batch["data"][:, :1].tile(1, 1, E, 1, 1)}
    total = 0.0
    for i, weight in enumerate(step_weights):
        if i % window == 0:
            x = {name: tensor.detach() for name, tensor in x.items()}
        y_pred = module(x, rollout_step=i)
        total = total + weight * _loss(y_pred, _target(batch, i))[0]
        x = dict(y_pred)
    total.backward()
    return total.detach(), _grads(module)


# ------------------------------------------------------------------------------------------------ the settings


def test_unset_windows_are_none() -> None:
    assert EnsembleTraining._validated_rollout_windows(None, 1) is None
    assert EnsembleTraining._validated_rollout_windows({}, 1) is None


def test_the_settings_are_checked() -> None:
    assert EnsembleTraining._validated_rollout_windows({"steps": 4, "weights": [3, 1]}, 1) == (4, (3.0, 1.0))
    with pytest.raises(ValueError, match="at least 1"):
        EnsembleTraining._validated_rollout_windows({"steps": 0}, 1)
    with pytest.raises(ValueError, match="non-negative"):
        EnsembleTraining._validated_rollout_windows({"steps": 2, "weights": [1, -1]}, 1)
    with pytest.raises(ValueError, match="accum_grad_batches"):
        EnsembleTraining._validated_rollout_windows({"steps": 2}, 2)


def test_each_step_takes_its_windows_share_of_the_weight() -> None:
    module = _module(4, (2, (3.0, 1.0)))
    assert module._rollout_window_step_weights(4) == pytest.approx([3 / 8, 3 / 8, 1 / 8, 1 / 8])
    # the plan's Stage B: 40 steps in windows of 4, the first window 3x the others = 25% of the loss
    module = _module(40, (4, (3.0,) + (1.0,) * 9))
    weights = module._rollout_window_step_weights(40)
    assert sum(weights[:4]) == pytest.approx(0.25)
    assert sum(weights) == pytest.approx(1.0)


def test_equal_or_no_weights_give_the_mean_over_steps() -> None:
    module = _module(6, (3, ()))
    assert module._rollout_window_step_weights(6) == pytest.approx([1 / 6] * 6)


def test_the_rollout_must_be_whole_windows_and_have_a_weight_per_window() -> None:
    module = _module(6, (4, ()))
    with pytest.raises(ValueError, match="whole number"):
        module._rollout_window_step_weights(6)
    module = _module(8, (4, (1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="3 weights, for 2 windows"):
        module._rollout_window_step_weights(8)


# ------------------------------------------------------------------------------------------------ the gradients


def test_the_gradients_are_those_of_the_weighted_loss_cut_at_the_window_boundaries(wired: Wired) -> None:
    batch = _batch(6)
    module = wired(6, (2, (3.0, 1.0, 2.0)))
    total, grads, calls = _windowed(module, batch)
    assert calls == [False, False, True]  # one backward pass per window; only the last one syncs

    reference = wired(6, None)
    expected_total, expected_grads = _reference(reference, batch, 2, module._rollout_window_step_weights(6))
    torch.testing.assert_close(total, expected_total)
    for grad, expected in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(grad, expected)


def test_the_cut_changes_the_gradients(wired: Wired) -> None:
    """Without the cut, the gradient would also flow back through the earlier windows' predictions."""
    batch = _batch(4)
    module = wired(4, (2, ()))
    _, cut, _ = _windowed(module, batch)
    uncut = wired(4, None)
    _, through = _reference(uncut, batch, 4, [0.25] * 4)
    assert not all(torch.allclose(a, b) for a, b in zip(cut, through, strict=True))


def test_one_window_is_the_usual_rollout(wired: Wired) -> None:
    batch = _batch(4)
    module = wired(4, (4, ()))
    total, grads, calls = _windowed(module, batch)
    assert calls == [True]

    usual = wired(4, None)
    loss = usual._step(batch).loss
    loss.backward()
    torch.testing.assert_close(total, loss.detach())
    for grad, expected in zip(grads, _grads(usual), strict=True):
        torch.testing.assert_close(grad, expected)


def test_the_steps_keep_their_absolute_index(wired: Wired) -> None:
    """The noise follows the lead time, so the 5th step is step 4 whichever window it is in."""
    module = wired(6, (2, ()))
    _windowed(module, _batch(6))
    assert module.model.fcsteps == [0, 1, 2, 3, 4, 5]


def test_validation_weighs_the_steps_as_training_does(wired: Wired) -> None:
    module = wired(4, (2, (3.0, 1.0)))
    losses = [torch.tensor(float(i + 1)) for i in range(6)]  # validation may unroll more than the 4 trained steps
    output = module._combine_loss_and_metrics(losses, [{} for _ in losses], [])
    assert output.loss.item() == pytest.approx(3 / 8 * 1 + 3 / 8 * 2 + 1 / 8 * 3 + 1 / 8 * 4)

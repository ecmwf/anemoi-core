# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DAForecaster: step policy, observation blend, metadata zeroing and residual base."""

from itertools import pairwise

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import Batch
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import DAForecaster
from anemoi.training.tasks import Forecaster
from anemoi.training.utils.masks import NoOutputMask
from tests.batch_builders import build_batch

# data order: a, b prognostic; f forcing; c corrector (observation metadata)
NAME_TO_INDEX = {"a": 0, "b": 1, "f": 2, "c": 3}
GRIDDED = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR = TensorLayout(ensemble=0, grid=1, variables=2)
GRID = 5
MASK = {"data": NoOutputMask()}


def _indices() -> IndexCollection:
    return IndexCollection(OmegaConf.create({"forcing": ["f"], "diagnostic": [], "corrector": ["c"]}), NAME_TO_INDEX)


def _task(**kwargs) -> DAForecaster:
    return DAForecaster(multistep_input=2, multistep_output=1, timestep="6H", **kwargs)


def _gridded(data: torch.Tensor, variables: list[str]) -> Batch:
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(GRID, 2)},
        layouts={"data": GRIDDED},
        variables={"data": variables},
        statistics={"data": {"mean": np.zeros(len(variables)), "stdev": np.ones(len(variables))}},
    )


# -- step policy -----------------------------------------------------------------------------


def test_steps_tag_da_cycles_before_the_rollout() -> None:
    task = _task(da_cycles=2, rollout={"start": 2, "maximum": 3}, validation_rollout=3)
    assert task.steps("training") == (
        {"rollout_step": 0, "is_da": True},
        {"rollout_step": 1, "is_da": True},
        {"rollout_step": 2, "is_da": False},
        {"rollout_step": 3, "is_da": False},
    )
    assert len(task.steps("validation")) == 2 + 3
    assert task.num_steps == 4
    assert task.num_loss_steps == 2


def test_offsets_cover_the_da_cycles_and_stay_a_prefix() -> None:
    task = _task(da_cycles=2, rollout={"start": 1, "maximum": 3})
    longest = task.get_offsets()
    assert len(task.get_offsets("training")) == 2 + 2 + 1
    assert len(longest) == 2 + 2 + 3
    # batch indices are taken against the longest offsets, so the training ones must be its prefix
    assert longest[: len(task.get_offsets("training"))] == task.get_offsets("training")
    assert [task.get_batch_output_indices(**step) for step in task.steps()] == [[2], [3], [4]]


def test_zero_cycles_behaves_as_the_forecaster() -> None:
    task = _task(rollout={"start": 2, "maximum": 3}, validation_rollout=4)
    forecaster = Forecaster(2, 1, "6H", rollout={"start": 2, "maximum": 3}, validation_rollout=4)
    for mode in ("training", "validation", None):
        assert task.get_offsets(mode) == forecaster.get_offsets(mode)
    assert task.num_loss_steps == forecaster.num_loss_steps
    for step in task.steps("validation"):
        assert task.get_metric_name(**step) == forecaster.get_metric_name(**step)
        assert task.get_metric_step(**step) == forecaster.get_metric_step(**step)
        assert task.step_loss_weight(**step) == 1.0
    assert not task.requires_skip_input


def test_metric_names_labels_and_weights() -> None:
    task = _task(da_cycles=2, da_loss_weight=0.25, checkpoint_steps=3, rollout={"start": 2, "maximum": 2})
    steps = task.steps()
    assert [task.get_metric_name(**s) for s in steps] == ["_dacycle0", "_dacycle1", "_rstep0", "_rstep1"]
    assert [task.get_step_label(**s) for s in steps] == ["dacycle00", "dacycle01", "rstep00", "rstep01"]
    # forecast steps keep the plain /1, /2 metric suffixes of a forecaster
    assert [task.get_metric_step(**s) for s in steps] == ["da1", "da2", 0, 1]
    assert [task.step_loss_weight(**s) for s in steps] == [0.25, 0.25, 1.0, 1.0]
    assert [task.step_uses_checkpoint(**s) for s in steps] == [True, True, True, False]
    assert [task.advance_needs_observations(**s) for s in steps] == [True, True, False, False]


@pytest.mark.parametrize("kwargs", [{"da_cycles": -1}, {"da_loss_weight": -0.1}, {"checkpoint_steps": 4}])
def test_invalid_arguments_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be"):
        _task(**{"da_cycles": 2, **kwargs})


# -- state advance ---------------------------------------------------------------------------


def _gridded_state() -> tuple[DAForecaster, Batch, Batch, Batch, Batch]:
    """Input (2 steps), prediction, imputed truth and NaN-carrying observations of one gridded step."""
    task = _task(da_cycles=1, rollout={"start": 1, "maximum": 1})
    x = _gridded(torch.zeros(1, 2, 1, GRID, 4), list(NAME_TO_INDEX))
    x = x.replace("data", x["data"].clone(data=x["data"].data + torch.arange(2.0).view(1, 2, 1, 1, 1)))
    y_pred = _gridded(torch.full((1, 1, 1, GRID, 2), 10.0), ["a", "b"])
    observed = torch.full((1, 1, 1, GRID, 4), 20.0)
    observed[..., NAME_TO_INDEX["f"]] = 30.0
    observed[..., NAME_TO_INDEX["c"]] = 40.0
    observed[0, 0, 0, :2, NAME_TO_INDEX["a"]] = torch.nan  # missing observations
    observed = _gridded(observed, list(NAME_TO_INDEX))
    truth = observed.replace("data", observed["data"].map_data(torch.nan_to_num))  # imputed
    return task, x, y_pred, truth, observed


def test_da_blend_uses_observations_where_present_background_where_missing() -> None:
    task, x, y_pred, truth, observed = _gridded_state()
    out = task.advance_input(
        x,
        y_pred,
        truth,
        rollout_step=0,
        is_da=True,
        data_indices={"data": _indices()},
        output_mask=MASK,
        observed=observed,
    )
    new = out["data"].data[0, -1, 0]  # the input window that comes from the output
    model_index = _indices().model.input.name_to_index
    assert torch.equal(new[:2, model_index["a"]], torch.full((2,), 10.0))  # background where missing
    assert torch.equal(new[2:, model_index["a"]], torch.full((3,), 20.0))  # observation where present
    assert torch.equal(new[:, model_index["b"]], torch.full((GRID,), 20.0))
    assert torch.equal(new[:, model_index["f"]], torch.full((GRID,), 30.0))
    assert torch.equal(new[:, model_index["c"]], torch.full((GRID,), 40.0))  # observed metadata
    # the previous last input window moved to the front
    assert torch.equal(out["data"].data[0, 0], x["data"].data[0, 1])


def test_forecast_advance_zeroes_the_metadata() -> None:
    task, x, y_pred, truth, _ = _gridded_state()
    out = task.advance_input(
        x,
        y_pred,
        truth,
        rollout_step=1,
        is_da=False,
        data_indices={"data": _indices()},
        output_mask=MASK,
    )
    new = out["data"].data[0, -1, 0]
    model_index = _indices().model.input.name_to_index
    assert torch.equal(new[:, model_index["a"]], torch.full((GRID,), 10.0))  # prediction
    assert torch.equal(new[:, model_index["f"]], torch.full((GRID,), 30.0))  # forcing from the truth
    assert torch.equal(new[:, model_index["c"]], torch.zeros(GRID))


def test_da_cycle_needs_the_observations() -> None:
    task, x, y_pred, truth, _ = _gridded_state()
    with pytest.raises(ValueError, match="observed"):
        task.advance_input(x, y_pred, truth, rollout_step=0, is_da=True, data_indices={"data": _indices()})


def _tabular(windows: list[torch.Tensor], variables: list[str], timedeltas: list[float]) -> TabularSource:
    starts = np.cumsum([0, *[w.shape[0] for w in windows]])
    return TabularSource(
        name="obs",
        variables=variables,
        layout=TABULAR,
        statistics={"mean": np.zeros(len(variables)), "stdev": np.ones(len(variables))},
        data=[torch.cat(windows)[None]],
        coordinates=[torch.zeros(int(starts[-1]), 2)],
        timedeltas=[torch.cat([torch.full((w.shape[0],), t) for w, t in zip(windows, timedeltas, strict=True)])],
        boundaries=[tuple(slice(int(a), int(b)) for a, b in pairwise(starts))],
    )


@pytest.mark.parametrize("is_da", [True, False])
def test_tabular_advance(is_da: bool) -> None:
    task = _task(da_cycles=1, rollout={"start": 1, "maximum": 1})
    indices = {"obs": _indices()}
    x = Batch({"obs": _tabular([torch.zeros(3, 4), torch.ones(2, 4)], list(NAME_TO_INDEX), [-21600.0, 0.0])})
    observed = torch.tensor([[20.0, 21.0, 30.0, 40.0], [torch.nan, 22.0, 31.0, 41.0]])
    observed = Batch({"obs": _tabular([observed], list(NAME_TO_INDEX), [21600.0])})
    truth = Batch({"obs": observed["obs"].map_data(torch.nan_to_num)})
    y_pred = Batch({"obs": _tabular([torch.full((2, 2), 10.0)], ["a", "b"], [21600.0])})

    out = task.advance_input(
        x,
        y_pred,
        truth,
        rollout_step=0,
        is_da=is_da,
        data_indices=indices,
        **({"observed": observed} if is_da else {}),
    )
    new = out["obs"].select_time([1]).data[0][0]
    if is_da:
        expected = torch.tensor([[20.0, 21.0, 30.0, 40.0], [10.0, 22.0, 31.0, 41.0]])
    else:
        expected = torch.tensor([[10.0, 10.0, 30.0, 0.0], [10.0, 10.0, 31.0, 0.0]])
    assert torch.equal(new, expected)


# -- residual base ---------------------------------------------------------------------------


def test_skip_input_is_off_by_default_and_without_cycles() -> None:
    _, x, _, _, _ = _gridded_state()
    assert _task(da_cycles=1).build_skip_input(x) is None
    assert _task(da_flow_dependent_skip=True).build_skip_input(x) is None


def test_skip_input_zero_base_then_background_after_a_blend() -> None:
    task, x, y_pred, truth, observed = _gridded_state()
    task.da_flow_dependent_skip = True
    indices = {"data": _indices()}

    first = task.build_skip_input(x, data_indices=indices)
    assert torch.equal(first["data"].data, torch.zeros_like(x["data"].data))

    advanced = task.advance_input(
        x,
        y_pred,
        truth,
        rollout_step=0,
        is_da=True,
        data_indices=indices,
        output_mask=MASK,
        observed=observed,
    )
    base = task.build_skip_input(advanced, y_pred, data_indices=indices, rollout_step=0, is_da=True)
    model_index = indices["data"].model.input.name_to_index
    # the prognostics of the new window are the raw background, everything else is the advanced input
    assert torch.equal(base["data"].data[0, -1, 0, :, model_index["a"]], torch.full((GRID,), 10.0))
    assert torch.equal(base["data"].data[0, -1, 0, :, model_index["c"]], torch.full((GRID,), 40.0))
    assert torch.equal(base["data"].data[:, 0], advanced["data"].data[:, 0])

    assert task.build_skip_input(advanced, y_pred, data_indices=indices, rollout_step=1, is_da=False) is None

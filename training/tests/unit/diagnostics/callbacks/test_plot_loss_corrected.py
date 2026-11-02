# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""PlotLossCorrected scores the corrected predictions of the forecast steps."""

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import Batch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot import LossCurvePlot
from anemoi.training.diagnostics.callbacks.plot import PlotLossCorrected
from anemoi.training.diagnostics.evaluation.plotting.loss import loss_contribution_plot_fn
from anemoi.training.tasks import DAForecaster
from anemoi.training.train.corrector import InstrumentCorrectors
from tests.batch_builders import build_batch

NAME_TO_INDEX = {"a": 0, "b": 1, "f": 2, "c": 3}  # a, b prognostic; f forcing; c corrector
GRID = 4


def _indices() -> IndexCollection:
    return IndexCollection(OmegaConf.create({"forcing": ["f"], "diagnostic": [], "corrector": ["c"]}), NAME_TO_INDEX)


def _gridded(data: torch.Tensor, variables: list[str]) -> Batch:
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(GRID, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": variables},
        statistics={"data": {"mean": np.zeros(len(variables)), "stdev": np.ones(len(variables))}},
    )


def _module(with_corrector: bool = True) -> SimpleNamespace:
    corrector = InstrumentCorrectors(
        instrument_groups={"obs": {"corrector_variables": ["c"], "channels": ["a"]}},
        all_corrector_names=["c"],
        output_name_to_index=_indices().model.output.name_to_index,
        hidden_dim=4,
    )
    corrector.correctors["obs"].out.bias.data.fill_(0.5)  # a correction of +0.5 on "a"
    return SimpleNamespace(
        corrector=torch.nn.ModuleDict({"data": corrector}) if with_corrector else torch.nn.ModuleDict(),
        task=DAForecaster(2, 1, "6H", rollout={"start": 1, "maximum": 1}, da_cycles=1),
        data_indices={"data": _indices()},
        preprocess_targets=lambda batch: batch,
    )


def _predictions() -> list[dict]:
    """One prediction per validation step: a DA cycle, then a forecast step."""
    return [dict(_gridded(torch.zeros(1, 1, 1, GRID, 2), ["a", "b"]).items()) for _ in range(2)]


def test_defaults() -> None:
    callback = PlotLossCorrected(parameter_groups={})
    assert callback.plot_fn is loss_contribution_plot_fn
    assert callback.tag_prefix == "loss_corrected"
    assert LossCurvePlot(parameter_groups={}).tag_prefix == "loss"


def test_only_forecast_steps_are_plotted() -> None:
    module = _module()
    assert [i for i, _ in PlotLossCorrected(parameter_groups={})._plotted_steps(module)] == [1]
    assert [i for i, _ in LossCurvePlot(parameter_groups={})._plotted_steps(module)] == [0, 1]


def test_forecast_predictions_are_corrected() -> None:
    module = _module()
    batch = _gridded(torch.ones(1, 4, 1, GRID, 4), list(NAME_TO_INDEX))
    adjusted = PlotLossCorrected(parameter_groups={})._adjust_predictions(module, _predictions(), batch)
    # the DA cycle is untouched; the forecast step is corrected on "a" only
    assert torch.equal(adjusted[0]["data"].data, torch.zeros(1, 1, 1, GRID, 2))
    assert torch.allclose(adjusted[1]["data"].data[..., 0], torch.full((1, 1, 1, GRID), 0.5))
    assert torch.equal(adjusted[1]["data"].data[..., 1], torch.zeros(1, 1, 1, GRID))


def test_without_corrector_it_is_the_loss_curve_plot() -> None:
    predictions = _predictions()
    batch = _gridded(torch.ones(1, 4, 1, GRID, 4), list(NAME_TO_INDEX))
    adjusted = PlotLossCorrected(parameter_groups={})._adjust_predictions(_module(False), predictions, batch)
    assert adjusted is predictions

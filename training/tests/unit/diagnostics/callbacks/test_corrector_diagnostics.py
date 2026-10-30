# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CorrectorDiagnostics: a smoke test with an MLP corrector, a DA forecaster and AdamW."""

import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.corrector import CorrectorDiagnostics
from anemoi.training.diagnostics.callbacks.corrector import _availability
from anemoi.training.tasks import DAForecaster
from anemoi.training.train.corrector import InstrumentCorrectors
from tests.batch_builders import build_batch

NAME_TO_INDEX = {"hirs_1": 0, "hirs_2": 1, "geom": 2, "f": 3}
GRID = 8


def _indices() -> IndexCollection:
    return IndexCollection(OmegaConf.create({"forcing": ["f"], "diagnostic": [], "corrector": ["geom"]}), NAME_TO_INDEX)


class _Logger:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def log_metrics(self, metrics: dict, step: int) -> None:
        del step
        self.logged.append(metrics)


def _setup() -> tuple[CorrectorDiagnostics, types.SimpleNamespace, types.SimpleNamespace, InstrumentCorrectors]:
    corrector = InstrumentCorrectors(
        instrument_groups={"hirs": {"corrector_variables": ["geom"], "channels": None}},
        all_corrector_names=["geom"],
        output_name_to_index=_indices().model.output.name_to_index,
        hidden_dim=4,
    )
    task = DAForecaster(2, 1, "6H", rollout={"start": 1, "maximum": 1}, da_cycles=1, da_loss_weight=0.0)
    pl_module = types.SimpleNamespace(
        corrector=torch.nn.ModuleDict({"data": corrector}),
        data_indices={"data": _indices()},
        task=task,
        model=torch.nn.Linear(2, 2),
    )
    trainer = types.SimpleNamespace(
        global_step=0,
        global_rank=0,
        loggers=[_Logger()],
        strategy=types.SimpleNamespace(root_device=torch.device("cpu")),
    )
    callback = CorrectorDiagnostics(dataset_name="data", every_n_steps=1)
    callback.setup(trainer, pl_module, "fit")
    return callback, pl_module, trainer, corrector


def _batch() -> object:
    data = torch.randn(1, 4, 1, GRID, len(NAME_TO_INDEX))
    data[:, 3, :, : GRID // 2, NAME_TO_INDEX["hirs_1"]] = torch.nan  # half the points miss hirs_1 in the forecast
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(GRID, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": list(NAME_TO_INDEX)},
        statistics={"data": {"mean": np.zeros(4), "stdev": np.ones(4)}},
    )


def test_logs_availability_activations_and_optimizer_state() -> None:
    callback, pl_module, trainer, corrector = _setup()
    params = [*corrector.parameters(), *pl_module.model.parameters()]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for global_step in range(2):  # the second step has AdamW moments to report
        trainer.global_step = global_step
        callback.on_train_batch_start(trainer, pl_module, _batch(), 0)
        pred = torch.randn(1, 1, 1, GRID, 2, requires_grad=True)
        out = corrector(pred, torch.randn(1, 1, 1, GRID, 1))
        (out.sum() + pl_module.model(torch.ones(2)).sum()).backward()
        callback.on_before_optimizer_step(trainer, pl_module, optimizer)
        optimizer.step()
        optimizer.zero_grad()

    metrics = trainer.loggers[0].logged[-1]
    # only the forecast step carries loss (da_loss_weight=0); there, half the points miss hirs_1
    assert metrics["corrector_diag/hirs/obs_frac"] == pytest.approx(0.75)
    assert metrics["corrector_diag/hirs/obs_point_frac"] == pytest.approx(1.0)
    assert metrics["corrector_diag/hirs/obs_sample_frac"] == pytest.approx(1.0)
    assert metrics["corrector_diag/hirs/meta_nonzero_frac"] == pytest.approx(1.0)
    # zero-initialised head, moved by one optimizer step
    assert 0 < metrics["corrector_diag/hirs/correction_rms"] < 0.1
    for key in ("grad_norm", "param_rms", "upd_fast_rms", "upd_slow_rms", "sqrt_v_floor_frac"):
        assert f"corrector_diag/hirs/{key}" in metrics
        assert f"corrector_diag/_model/{key}" in metrics


def test_inactive_between_logging_steps_and_without_corrector() -> None:
    callback, pl_module, trainer, _ = _setup()
    callback.every_n_steps = 5
    trainer.global_step = 3
    callback.on_train_batch_start(trainer, pl_module, _batch(), 0)
    callback.on_before_optimizer_step(trainer, pl_module, torch.optim.SGD(pl_module.model.parameters(), lr=0.1))
    assert not trainer.loggers[0].logged

    other = CorrectorDiagnostics(dataset_name="obs")
    other.setup(trainer, pl_module, "fit")
    assert not other._keys
    callback.teardown(trainer, pl_module, "fit")
    assert not callback._handles


def test_availability_of_tabular_targets() -> None:
    source = TabularSource(
        name="obs",
        variables=list(NAME_TO_INDEX),
        layout=TensorLayout(ensemble=0, grid=1, variables=2),
        statistics={},
        data=[
            torch.tensor([[[1.0, torch.nan, 0.0, 0.0], [torch.nan, torch.nan, 0.0, 0.0]]]),
            torch.full((1, 1, 4), torch.nan),
        ],
        coordinates=[torch.zeros(2, 2), torch.zeros(1, 2)],
        timedeltas=[torch.zeros(2), torch.zeros(1)],
        boundaries=[(slice(0, 2),), (slice(0, 1),)],
    )
    assert _availability(source, ["hirs_1", "hirs_2"]) == {
        "obs_frac": pytest.approx(1 / 6),
        "obs_point_frac": pytest.approx(1 / 3),
        "obs_sample_frac": pytest.approx(1 / 2),
    }

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""LossShareMonitor: grouping, leaf selection and the recorded windows on Source-based losses."""

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import Source
from anemoi.models.data.sources import TabularSource
from anemoi.training.diagnostics.callbacks.loss_shares import LossShareMonitor
from anemoi.training.diagnostics.callbacks.loss_shares import leaf_weights
from anemoi.training.diagnostics.callbacks.loss_shares import variable_group
from anemoi.training.losses import MSELoss
from anemoi.training.losses.combined import CombinedLoss
from tests.batch_builders import build_batch

VARIABLES = ["t_850", "t_500", "2t", "iasi_921"]


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("t_850", "t"),
        ("q_1000", "q"),
        ("2t", "surface"),
        ("msl", "surface"),
        ("cf", "cf"),
        ("cris_cf", "cris_cf"),
        ("iasi_921", "iasi"),
        ("georing_vis064", "georing"),
        ("t_surface", "t"),
        ("z", "z"),
    ],
)
def test_default_groups(name: str, group: str) -> None:
    assert variable_group(name) == group


def test_configured_groups() -> None:
    groups = {"sondes": ["t_*", "q_*"], "radiances": ["iasi_*", "cris_*"]}
    assert variable_group("t_850", groups) == "sondes"
    assert variable_group("cris_cf", groups) == "radiances"  # explicit groups win over the single variables
    assert variable_group("2t", groups) == "surface"
    assert variable_group("w_850", groups, pressure_level_params=("z",)) == "w"
    monitor = LossShareMonitor(".", surface_variables=("2t",), per_variable=False)
    assert monitor.group_of("msl") == "msl"
    assert LossShareMonitor(".", per_variable=True).group_of("t_850") == "t_850"


def _combined() -> CombinedLoss:
    return CombinedLoss(
        losses=[{"_target_": "anemoi.training.losses.MSELoss"}, {"_target_": "anemoi.training.losses.MAELoss"}],
        loss_weights=(1.0, 0.5),
    )


def test_leaf_selection_and_weights() -> None:
    combined = _combined()
    leaves = list(combined.iter_leaf_losses())
    assert leaf_weights(combined) == [1.0, 0.5]
    assert leaf_weights(MSELoss()) == [1.0]
    assert LossShareMonitor(".")._select_leaf([MSELoss()]) == 0
    assert LossShareMonitor(".", leaf=1)._select_leaf(leaves) == 1
    assert LossShareMonitor(".", leaf=-1)._select_leaf(leaves) == 1
    assert LossShareMonitor(".", leaf="MAELoss")._select_leaf(leaves) == 1
    assert LossShareMonitor(".", leaf=leaves[0].name)._select_leaf(leaves) == 0
    with pytest.raises(ValueError, match="several leaves"):
        LossShareMonitor(".")._select_leaf(leaves)
    with pytest.raises(ValueError, match="matches 0"):
        LossShareMonitor(".", leaf="KCRPS")._select_leaf(leaves)
    with pytest.raises(ValueError, match="out of range"):
        LossShareMonitor(".", leaf=2)._select_leaf(leaves)


def _gridded_pair() -> tuple[Source, Source]:
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 1, 1, 7, len(VARIABLES), generator=generator)
    batch = build_batch(
        data={"d": pred},
        coordinates={"d": torch.zeros(7, 2)},
        layouts={"d": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"d": VARIABLES},
    )
    source = batch["d"]
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return source, source.map_data(lambda t: t + scale)


def _tabular_pair() -> tuple[Source, Source]:
    def source(data: list[torch.Tensor]) -> TabularSource:
        return TabularSource(
            name="d",
            variables=VARIABLES,
            layout=TensorLayout(ensemble=0, grid=1, variables=2),
            statistics={},
            data=data,
            coordinates=[torch.zeros(d.shape[1], 2) for d in data],
            timedeltas=[torch.zeros(d.shape[1]) for d in data],
            boundaries=[(slice(0, d.shape[1]),) for d in data],
        )

    pred = [torch.zeros(1, 5, 4), torch.zeros(1, 3, 4)]
    return source(pred), source([p + torch.tensor([1.0, 2.0, 3.0, 4.0]) for p in pred])


class _Logger:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int]] = []

    def log_metrics(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))


def _trainer(global_step: int = 0) -> types.SimpleNamespace:
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-3}])
    return types.SimpleNamespace(
        global_step=global_step,
        global_rank=0,
        current_epoch=0,
        sanity_checking=False,
        loggers=[_Logger()],
        optimizers=[optimizer],
    )


def _pl_module(loss: torch.nn.Module) -> types.SimpleNamespace:
    module = types.SimpleNamespace(loss={"d": loss}, device=torch.device("cpu"))

    def compute_loss_metrics(pred: Source, target: Source, **_kwargs) -> torch.Tensor:
        return module.loss["d"](pred, target)

    module.compute_loss_metrics = compute_loss_metrics
    return module


@pytest.mark.parametrize("pair", [_gridded_pair, _tabular_pair], ids=["gridded", "tabular"])
def test_window_records_and_logs_shares(pair, tmp_path: Path) -> None:  # noqa: ANN001
    pred, target = pair()
    module = _pl_module(_combined())
    monitor = LossShareMonitor(str(tmp_path), dataset_name="d", every_n_steps=2, leaf="MSELoss")
    trainer = _trainer()
    monitor.setup(trainer, module, "fit")

    for global_step, kwargs in [(1, {"rollout_step": 0, "is_da": True}), (2, {"rollout_step": 1, "is_da": False})]:
        trainer.global_step = global_step - 1
        monitor.on_train_batch_start(trainer, module, None, 0)
        module.compute_loss_metrics(pred, target, **kwargs)
        trainer.global_step = global_step
        monitor.on_train_batch_end(trainer, module, None, None, 0)

    (window,) = list(tmp_path.iterdir())
    raw = np.load(window / "loss_shares_raw.npz")
    meta = json.loads((window / "meta.json").read_text())
    assert list(raw["names"]) == VARIABLES
    assert raw["full"].shape == (2, 4)
    assert list(raw["is_da"]) == [True, False]
    assert list(raw["step"]) == [0, 1]
    assert raw["other0"].shape == (2,)  # the MAE leaf, recorded as a scalar
    assert meta["leaf"] == "MSELoss"
    assert meta["other_leaves"] == [{"name": "MAELoss", "weight": 0.5}]
    assert meta["n_da_rows"] == 1
    np.testing.assert_allclose(meta["mean_step_loss"], raw["ret"].mean() + 0.5 * raw["other0"].mean(), rtol=1e-6)
    # the leaf's loss is the mean of its per-variable vector
    np.testing.assert_allclose(raw["full"].mean(1), raw["ret"], rtol=1e-5)

    ((metrics, step),) = trainer.loggers[0].logged
    assert step == 1
    assert set(metrics) == {"loss_share/train/t", "loss_share/train/surface", "loss_share/train/iasi"}
    assert sum(metrics.values()) == pytest.approx(1.0)
    # squared errors 1, 4, 9, 16 per variable (any grid normalisation cancels in the shares)
    assert metrics["loss_share/train/iasi"] == pytest.approx(16 / 30, rel=1e-5)


def test_recompute_in_backward_is_not_recorded(tmp_path: Path) -> None:
    pred, target = _gridded_pair()
    weight = torch.nn.Parameter(torch.tensor(1.0))
    module = _pl_module(MSELoss())
    monitor = LossShareMonitor(str(tmp_path), dataset_name="d", every_n_steps=100)
    trainer = _trainer()
    monitor.setup(trainer, module, "fit")
    monitor.on_train_batch_start(trainer, module, None, 0)

    def loss_fn(w: torch.Tensor) -> torch.Tensor:
        return module.compute_loss_metrics(pred.map_data(lambda t: t * w), target, rollout_step=0)

    checkpoint(loss_fn, weight, use_reentrant=False).backward()
    assert len(monitor._rows["train"]) == 1


def test_validation_outside_sanity_check_only(tmp_path: Path) -> None:
    pred, target = _gridded_pair()
    module = _pl_module(MSELoss())
    monitor = LossShareMonitor(str(tmp_path), dataset_name="d")
    trainer = _trainer()
    monitor.setup(trainer, module, "fit")

    trainer.sanity_checking = True
    monitor.on_validation_batch_start(trainer, module, None, 0)
    module.compute_loss_metrics(pred, target, rollout_step=0)
    assert not monitor._rows

    trainer.sanity_checking = False
    monitor.on_validation_batch_start(trainer, module, None, 0)
    module.compute_loss_metrics(pred, target, rollout_step=0)
    monitor.on_validation_batch_end(trainer, module, None, None, 0)
    monitor.on_validation_epoch_end(trainer, module)
    assert [p.name for p in tmp_path.iterdir()] == ["valid_s000000-000000"]

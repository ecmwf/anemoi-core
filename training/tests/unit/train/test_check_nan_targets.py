# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``BaseTrainingModule._check_nan_targets``: which losses count as unable to handle NaN targets."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses.aggregate import TimeAggregateLossWrapper
from anemoi.training.train.methods.base import BaseTrainingModule
from tests.batch_builders import build_batch

VARIABLES = ["a", "b"]


def _module() -> SimpleNamespace:
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {name: index for index, name in enumerate(VARIABLES)},
    )
    return SimpleNamespace(data_indices={"data": data_indices})


def _targets_with_nans() -> object:
    data = torch.ones(1, 1, 1, 4, len(VARIABLES))
    data[..., 0, 1] = float("nan")  # one missing value of "b"
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(4, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": VARIABLES},
    )["data"]


def _check(loss: torch.nn.Module) -> None:
    BaseTrainingModule._check_nan_targets(_module(), _targets_with_nans(), "data", {"training_loss": loss})


def _combined(*ignore_nans: bool) -> CombinedLoss:
    # A combined loss configured with ignore_nans: True, as ensemble_multiscale_aggregation.yaml does.
    return CombinedLoss(
        losses=[
            {"_target_": "anemoi.training.losses.MSELoss", "scalers": [], "ignore_nans": flag} for flag in ignore_nans
        ],
        ignore_nans=True,
    )


def test_combined_loss_whose_children_ignore_nans_passes() -> None:
    """The container's own ignore_nans is unused; only the losses it combines compute anything."""
    loss = _combined(True, True)
    assert loss.ignore_nans is False, "precondition: CombinedLoss keeps its own flag at the default"
    _check(loss)


def test_combined_loss_with_one_child_not_ignoring_nans_raises() -> None:
    with pytest.raises(ValueError, match=r"targets of \['b'\] contain NaNs, but \['training_loss'\]"):
        _check(_combined(True, False))


def test_wrapper_is_judged_by_its_inner_loss() -> None:
    _check(TimeAggregateLossWrapper(["mean"], MAELoss(ignore_nans=True)))


@pytest.mark.parametrize(("ignore_nans", "raises"), [(True, False), (False, True)])
def test_plain_loss(ignore_nans: bool, raises: bool) -> None:
    if raises:
        with pytest.raises(ValueError, match="use ignore_nans=False"):
            _check(MSELoss(ignore_nans=ignore_nans))
    else:
        _check(MSELoss(ignore_nans=ignore_nans))


def test_non_loss_module_falls_back_to_its_submodules() -> None:
    """Metrics need not be BaseLoss; any submodule with ignore_nans=False still counts."""
    metric = torch.nn.Sequential(MSELoss(ignore_nans=False))
    with pytest.raises(ValueError, match="use ignore_nans=False"):
        _check(metric)

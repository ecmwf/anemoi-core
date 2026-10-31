# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``BaseTrainingModule._check_nan_targets`` and ``_check_nan_inputs``: when NaNs must stop training."""

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

VARIABLES = ["a", "b", "diag"]


def _module() -> SimpleNamespace:
    """The parts of a training module the checks use: data indices and the static helpers."""
    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": ["diag"], "target": []}),
        {name: index for index, name in enumerate(VARIABLES)},
    )
    return SimpleNamespace(
        data_indices={"data": data_indices},
        _nan_flags_per_variable=BaseTrainingModule._nan_flags_per_variable,
        _variables_with_nans=BaseTrainingModule._variables_with_nans,
    )


def _batch_with_nan_in(variable: str | None) -> object:
    data = torch.ones(1, 1, 1, 4, len(VARIABLES))
    if variable is not None:
        data[..., 0, VARIABLES.index(variable)] = float("nan")  # one missing value
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(4, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": VARIABLES},
    )


def _targets_with_nans() -> object:
    return _batch_with_nan_in("b")["data"]


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


# --- model inputs ---------------------------------------------------------------------------------


def test_inputs_without_nans_pass() -> None:
    BaseTrainingModule._check_nan_inputs(_module(), _batch_with_nan_in(None))


def test_nan_in_a_model_input_raises_and_names_it() -> None:
    """A NaN input makes the whole prediction NaN, which an ignore_nans loss reports as 0."""
    with pytest.raises(ValueError, match=r"Model inputs contain NaNs after preprocessing: \{'data': \['b'\]\}"):
        BaseTrainingModule._check_nan_inputs(_module(), _batch_with_nan_in("b"))


def test_nan_in_a_variable_the_model_does_not_read_passes() -> None:
    """Diagnostic variables are outputs only, so a NaN there never reaches the model."""
    BaseTrainingModule._check_nan_inputs(_module(), _batch_with_nan_in("diag"))


def test_preprocess_inputs_runs_the_check() -> None:
    module = _module()
    module.model = SimpleNamespace(pre_processors={})
    module._map_dataset_processors = lambda batch, _processors: batch
    module._check_nan_inputs = lambda batch: BaseTrainingModule._check_nan_inputs(module, batch)
    with pytest.raises(ValueError, match="Model inputs contain NaNs"):
        BaseTrainingModule.preprocess_inputs(module, _batch_with_nan_in("a"))

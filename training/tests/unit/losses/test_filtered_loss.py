# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import MSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


def test_instantiation_with_filtering() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    data_config = {"forcing": ["forcing"], "diagnostic": [], "target": ["imerg"]}
    name_to_index = {"tp": 0, "forcing": 1, "imerg": 2}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "predicted_variables": ["tp"],
                "target_variables": ["imerg"],
                "scalers": ["grid_uniform", "dynamic"],
            },
        ),
        scalers={
            "grid_uniform": (3, torch.ones(4)),
            "dynamic": (4, torch.tensor([2.0, 7.0, 11.0])),
        },
        data_indices=data_indices,
    )

    assert isinstance(loss, LossVariableMapper)
    assert IndexSpace.MODEL_OUTPUT in loss.predicted_indices_by_layout
    assert loss.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0]
    assert loss.target_indices_by_layout[IndexSpace.DATA_FULL] == [2]
    torch.testing.assert_close(loss.loss.scaler.tensors["dynamic"][1], torch.tensor([2.0]))

    pred = torch.ones((1, 1, 1, 4, 1))
    target = torch.zeros((1, 1, 1, 4, len(name_to_index)))
    target[..., 2] = 3.0

    loss_total = loss(
        pred,
        target,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    torch.testing.assert_close(loss_total, torch.tensor(32.0))

    loss.update_scaler("dynamic", torch.tensor([13.0, 17.0, 19.0]), override=True)
    torch.testing.assert_close(loss.loss.scaler.tensors["dynamic"][1], torch.tensor([13.0]))


def test_instantiation_with_filtering_requires_layout_kwargs() -> None:
    from anemoi.models.data_indices.collection import IndexCollection

    data_indices = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": ["imerg"]}),
        {"tp": 0, "imerg": 1},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "predicted_variables": ["tp"],
                "target_variables": ["imerg"],
            },
        ),
        data_indices=data_indices,
    )

    pred = torch.ones((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 1, 2, 2))

    with pytest.raises(ValueError, match="requires both 'pred_layout' and 'target_layout'"):
        loss(pred, target)


def test_print_variable_scaling() -> None:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.scalers.scalers import create_scalers
    from anemoi.training.losses.utils import print_variable_scaling
    from anemoi.utils.config import DotDict

    data_config = {"data": {"forcing": ["f1"], "target": [], "prognostic": ["f2"], "diagnostic": ["tp", "imerg"]}}
    name_to_index = {"tp": 0, "imerg": 1, "f1": 2, "f2": 3}
    data_indices = IndexCollection(DictConfig(data_config), name_to_index)
    metadata_extractor = ExtractVariableGroupAndLevel(
        DotDict(
            {
                "default": "sfc",
            },
        ),
    )
    scalers, _ = create_scalers(
        DotDict(
            {
                "general_variable": {
                    "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                    "weights": {
                        "default": 1,
                        "tp": 0.1,
                        "imerg": 100,
                        "f2": 0.5,
                    },
                },
            },
        ),
        data_indices=data_indices,
        metadata_extractor=metadata_extractor,
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": ["general_variable"],
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MAELoss",
                        "scalers": ["general_variable"],
                        "predicted_variables": ["tp"],
                        "target_variables": ["imerg"],
                    },
                ],
            },
        ),
        data_indices=data_indices,
        scalers=scalers,
    )
    scaling_dict = print_variable_scaling(loss, data_indices)
    assert "LossVariableMapper" in scaling_dict  # loss is filtered
    assert "tp" in scaling_dict["LossVariableMapper"]
    assert all(var not in scaling_dict["LossVariableMapper"] for var in data_indices.name_to_index if var != "tp")


def test_loss_variable_mapper_propagates_needs_shard_layout_info() -> None:
    loss = LossVariableMapper(
        loss=MultiscaleLossWrapper(
            per_scale_loss=MSELoss(),
            weights=[1.0],
            keep_batch_sharded=True,
        ),
    )

    assert loss.needs_shard_layout_info is True


def test_loss_variable_mapper_rejects_non_base_loss() -> None:
    with pytest.raises(TypeError, match="Expected BaseLoss"):
        LossVariableMapper(loss=object())  # type: ignore[arg-type]


# =============================================================================
# LossVariableMapper behaviour tests
#
# The tests below exercise LossVariableMapper directly, bypassing get_loss_function.
# They use a shared fixture (data_indices_forcing_gaps) that deliberately has
# multiple forcing variables interspersed between predicted variables, creating
# non-contiguous gaps in the DATA_FULL index space. This is the common layout
# and is the main source of
# off-by-one bugs in scaler slicing and target index resolution.
#
# _mapper() is a thin helper that constructs and initialises a LossVariableMapper
# without going through the full get_loss_function factory, so each test can focus
# on a single behaviour in isolation.
# =============================================================================


@pytest.fixture
def data_indices_forcing_gaps() -> IndexCollection:
    """var_0..1, f_0..1(forcing), var_2..3, f_2..3(forcing), var_4, imerg(target).

    model.output.full = [0,1,4,5,8]  |  data.output.full = [0,1,4,5,8,9]
    """
    return IndexCollection(
        DictConfig({"forcing": ["f_0", "f_1", "f_2", "f_3"], "diagnostic": [], "target": ["imerg"]}),
        {
            "var_0": 0,
            "var_1": 1,
            "f_0": 2,
            "f_1": 3,
            "var_2": 4,
            "var_3": 5,
            "f_2": 6,
            "f_3": 7,
            "var_4": 8,
            "imerg": 9,
        },
    )


def _mapper(data_indices: IndexCollection, predicted: list[str], target: list[str] | None = None) -> LossVariableMapper:
    w = LossVariableMapper(loss=MSELoss(), predicted_variables=predicted, target_variables=target or predicted)
    w.set_data_indices(data_indices)
    return w


# --- VARIABLE-axis scaler filtering ------------------------------------------


class TestVariableAxisScalerFiltering:

    def test_mixed_dim_scaler_filtered_to_predicted(self, data_indices_forcing_gaps: IndexCollection) -> None:
        """Multi-dim (BATCH, GRID, VARIABLE) scaler is sliced on the VARIABLE axis."""
        w = _mapper(data_indices_forcing_gaps, ["var_0"], ["imerg"])
        n_model_vars = len(data_indices_forcing_gaps.model.output.full)
        w.add_scaler(dimension=(0, 3, 4), scaler=torch.ones(2, 8, n_model_vars), name="mixed")

        assert w.loss.scaler.tensors["mixed"][1].shape == (2, 8, 1)

    def test_broadcast_variable_axis_untouched(self, data_indices_forcing_gaps: IndexCollection) -> None:
        """Broadcast VARIABLE axis (size 1) must not be index-selected."""
        w = _mapper(data_indices_forcing_gaps, ["var_0"], ["imerg"])
        w.add_scaler(dimension=(0, 3, 4), scaler=torch.ones(2, 8, 1), name="bc")

        assert w.loss.scaler.tensors["bc"][1].shape == (2, 8, 1)

    def test_data_full_scaler_with_forcing_gap(self) -> None:
        """DATA_FULL-sized scaler correctly resolves through forcing gaps."""
        di = IndexCollection(
            DictConfig({"forcing": ["f0"], "diagnostic": [], "target": []}),
            {"f0": 0, "var_0": 1, "var_1": 2},
        )
        w = _mapper(di, ["var_1"])
        w.add_scaler(dimension=4, scaler=torch.tensor([10.0, 20.0, 30.0]), name="full")

        # var_1 is at data-full index 2 → value 30.0
        torch.testing.assert_close(w.loss.scaler.tensors["full"][1], torch.tensor([30.0]))

    def test_update_scaler_refilters(self, data_indices_forcing_gaps: IndexCollection) -> None:
        """update_scaler with override re-applies VARIABLE-axis filtering."""
        w = _mapper(data_indices_forcing_gaps, ["var_2"])
        w.add_scaler(dimension=4, scaler=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="vw")
        torch.testing.assert_close(w.loss.scaler.tensors["vw"][1], torch.tensor([3.0]))

        w.update_scaler("vw", torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]), override=True)
        torch.testing.assert_close(w.loss.scaler.tensors["vw"][1], torch.tensor([30.0]))


# --- variable ordering and defaults ------------------------------------------


def test_scaler_preserves_reversed_variable_order() -> None:
    """Reversed predicted_variables → scaler values in matching reversed order."""
    di = IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": ["imerg"]}),
        {f"var_{i}": i for i in range(10)} | {"imerg": 10},
    )
    reversed_vars = [f"var_{i}" for i in range(9, -1, -1)]
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MSELoss",
                "predicted_variables": reversed_vars,
                "target_variables": reversed_vars,
                "scalers": ["pl"],
            },
        ),
        scalers={"pl": (4, torch.tensor([0.1 * (i + 1) for i in range(11)]))},
        data_indices=di,
    )

    expected = torch.tensor([0.1 * (i + 1) for i in range(9, -1, -1)])
    torch.testing.assert_close(loss.loss.scaler.tensors["pl"][1], expected)


def test_default_target_inherits_predicted_order(data_indices_forcing_gaps: IndexCollection) -> None:
    """target_variables=None copies predicted_variables list and resolves indices."""
    w = LossVariableMapper(loss=MSELoss(), predicted_variables=["var_3", "var_0"], target_variables=None)
    w.set_data_indices(data_indices_forcing_gaps)

    assert w.target_variables == ["var_3", "var_0"]
    assert w.target_indices_by_layout[IndexSpace.DATA_FULL] == [5, 0]


def test_constructor_can_eagerly_set_data_indices(data_indices_forcing_gaps: IndexCollection) -> None:
    w = LossVariableMapper(
        loss=MSELoss(),
        predicted_variables=["var_3", "var_0"],
        target_variables=None,
        data_indices=data_indices_forcing_gaps,
    )

    assert w.target_variables == ["var_3", "var_0"]
    assert w.target_indices_by_layout[IndexSpace.DATA_FULL] == [5, 0]


# --- layout validation -------------------------------------------------------


def test_rejects_unavailable_target_layout(data_indices_forcing_gaps: IndexCollection) -> None:
    """Target-only variable makes MODEL_OUTPUT unavailable as target_layout."""
    w = _mapper(data_indices_forcing_gaps, ["var_0"], ["imerg"])
    with pytest.raises(ValueError, match="target_layout 'model_output' is not available"):
        w(
            torch.zeros(1, 1, 1, 2, 6),
            torch.zeros(1, 1, 1, 2, 10),
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.MODEL_OUTPUT,
        )


def test_rejects_invalid_layout_string(data_indices_forcing_gaps: IndexCollection) -> None:
    """Invalid layout name raises ValueError with clear message."""
    w = _mapper(data_indices_forcing_gaps, ["var_0"], ["imerg"])
    with pytest.raises(ValueError, match="Invalid pred_layout"):
        w(
            torch.zeros(1, 1, 1, 2, 6),
            torch.zeros(1, 1, 1, 2, 10),
            pred_layout="not_a_layout",
            target_layout=IndexSpace.DATA_FULL,
        )


# --- scaler_indices remapping ------------------------------------------------


class TestScalerIndicesRemapping:

    def test_global_to_local_remap(self, data_indices_forcing_gaps: IndexCollection) -> None:
        """Global scaler_indices are translated to filtered-tensor-local positions."""
        w = _mapper(data_indices_forcing_gaps, ["var_0", "var_3"])
        w.add_scaler(dimension=3, scaler=torch.ones(8), name="grid")
        w.add_scaler(dimension=4, scaler=torch.tensor([2.0, 3.0]), name="var_w")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0  # var_0
        pred[..., 3] = 1.0  # var_3

        # Request global index 3 (var_3) → should remap to local index 1
        loss = w(
            pred,
            target,
            scaler_indices=(..., [3]),
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )

        # MSE=(1-0)²=1, var_w[1]=3, 8 grid points → 24
        torch.testing.assert_close(loss, torch.tensor(24.0))

    def test_empty_remap_returns_zero(self, data_indices_forcing_gaps: IndexCollection) -> None:
        """scaler_indices selecting no filtered variables → zero tensor."""
        w = _mapper(data_indices_forcing_gaps, ["var_0"])
        w.add_scaler(dimension=3, scaler=torch.ones(8), name="grid")
        w.add_scaler(dimension=4, scaler=torch.tensor([2.0]), name="var_w")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0

        # Global index 3 not in filtered set ["var_0"]
        loss = w(
            pred,
            target,
            scaler_indices=(..., [3]),
            pred_layout=IndexSpace.DATA_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_partial_remap_logs_debug(
        self,
        data_indices_forcing_gaps: IndexCollection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Partially dropped scaler indices should be visible at DEBUG level."""
        w = _mapper(data_indices_forcing_gaps, ["var_0", "var_3"])
        w.add_scaler(dimension=3, scaler=torch.ones(8), name="grid")
        w.add_scaler(dimension=4, scaler=torch.tensor([2.0, 3.0]), name="var_w")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0
        pred[..., 3] = 1.0

        with caplog.at_level(logging.DEBUG, logger="anemoi.training.losses.variable_mapper"):
            w(
                pred,
                target,
                scaler_indices=(..., [0, 5]),
                pred_layout=IndexSpace.DATA_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
            )

        assert "dropped scaler variable indices during filtering" in caplog.text

    def test_empty_remap_logs_warning(
        self,
        data_indices_forcing_gaps: IndexCollection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Fully dropped scaler indices should warn because forward returns zeros."""
        w = _mapper(data_indices_forcing_gaps, ["var_0"])
        w.add_scaler(dimension=3, scaler=torch.ones(8), name="grid")
        w.add_scaler(dimension=4, scaler=torch.tensor([2.0]), name="var_w")

        pred = torch.zeros(1, 1, 1, 8, 6)
        target = torch.zeros(1, 1, 1, 8, 10)
        pred[..., 0] = 1.0

        with caplog.at_level(logging.WARNING, logger="anemoi.training.losses.variable_mapper"):
            loss = w(
                pred,
                target,
                scaler_indices=(..., [3]),
                pred_layout=IndexSpace.DATA_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
            )

        torch.testing.assert_close(loss, torch.tensor(0.0))
        assert "Metric selection is empty; forward() will return zeros for this call." in caplog.text

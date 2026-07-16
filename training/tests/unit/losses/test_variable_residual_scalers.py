# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.scalers import NoResidualScaler
from anemoi.training.losses.scalers import StdevResidualScaler
from anemoi.training.losses.scalers import VarResidualScaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

NAME_TO_INDEX = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6, "d": 7}

# Same numeric fixtures as test_loss_scaling.py's tendency-scaler tests, but statistics_residuals
# is a FLAT mapping (no "lead_times" / per-step keying), matching the residual-normalisation
# contract: residual = target - interp(source) is never step-dependent.
STATISTICS = {"stdev": [0.0, 10.0, 10, 10, 7.0, 3.0, 1.0, 2.0, 3.5]}
STATISTICS_RESIDUALS = {"stdev": [0.0, 5, 5, 5, 4.0, 7.5, 8.6, 1, 10]}


def _make_config(scaler_config: dict) -> DictConfig:
    return DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "training": {
                "training_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": ["general_variable", "additional_scaler"],
                },
                "variable_groups": {
                    "default": "sfc",
                    "pl": ["y"],
                },
                "scalers": {
                    "builders": {
                        "additional_scaler": scaler_config,
                        "general_variable": {
                            "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                            "weights": {
                                "default": 1,
                                "z": 0.1,
                                "other": 100,
                                "y": 0.5,
                            },
                        },
                    },
                },
                "metrics": ["other", "y_850"],
            },
        },
    )


def _data_indices() -> IndexCollection:
    config = _make_config({"_target_": "anemoi.training.losses.scalers.NoResidualScaler"})
    return IndexCollection(data_config=config.data, name_to_index=NAME_TO_INDEX)


no_residual_scaler = {"_target_": "anemoi.training.losses.scalers.NoResidualScaler"}
std_dev_residual_scaler = {"_target_": "anemoi.training.losses.scalers.StdevResidualScaler"}
var_residual_scaler = {"_target_": "anemoi.training.losses.scalers.VarResidualScaler"}

# Non-prognostic (forcing/diagnostic) variables are left untouched by the residual scalers
# (scaling stays 1.0 before the general-variable weight is applied), mirroring the tendency
# scalers' convention: "x" is forcing (not part of the output), "q"/"z" are diagnostic.
expected_no_residual_scaling = torch.Tensor(
    [
        1 * 0.5,  # y_50
        1 * 0.5,  # y_500
        1 * 0.5,  # y_850
        1 * 1,  # q (diagnostic)
        1 * 0.1,  # z (diagnostic)
        1 * 100,  # other
        1 * 1,  # d
    ],
)

expected_stdev_residual_scaling = torch.Tensor(
    [
        (10.0 / 5.0) * 0.5,  # y_50
        (10.0 / 5.0) * 0.5,  # y_500
        (10.0 / 5.0) * 0.5,  # y_850
        1 * 1,  # q (diagnostic)
        1 * 0.1,  # z (diagnostic)
        (1 / 8.6) * 100,  # other
        1 * 2.0,  # d
    ],
)

expected_var_residual_scaling = torch.Tensor(
    [
        (10.0**2) / (5.0**2) * 0.5,  # y_50
        (10.0**2) / (5.0**2) * 0.5,  # y_500
        (10.0**2) / (5.0**2) * 0.5,  # y_850
        1,  # q (diagnostic)
        0.1,  # z (diagnostic)
        (1**2) / (8.6**2) * 100,  # other
        (2**2) / (1**2),  # d
    ],
)


@pytest.mark.parametrize(
    ("scaler_config", "expected_scaling"),
    [
        (no_residual_scaler, expected_no_residual_scaling),
        (std_dev_residual_scaler, expected_stdev_residual_scaling),
        (var_residual_scaler, expected_var_residual_scaling),
    ],
)
def test_variable_residual_scaling_vals(
    scaler_config: dict,
    expected_scaling: torch.Tensor,
    graph_with_nodes: HeteroData,
) -> None:
    config = _make_config(scaler_config)
    data_indices = IndexCollection(data_config=config.data, name_to_index=NAME_TO_INDEX)
    metadata_extractor = ExtractVariableGroupAndLevel(config.training.variable_groups)

    scalers, _ = create_scalers(
        config.training.scalers.builders,
        data_indices=data_indices,
        graph_data=graph_with_nodes,
        statistics=STATISTICS,
        statistics_residuals=STATISTICS_RESIDUALS,
        metadata_extractor=metadata_extractor,
        output_mask=NoOutputMask(),
    )

    loss = get_loss_function(config.training.training_loss, scalers=scalers)

    final_variable_scaling = loss.scaler.subset_by_dim(TensorDim.VARIABLE.value).get_scaler(len(TensorDim))

    assert torch.allclose(final_variable_scaling, expected_scaling)


def test_no_residual_scaler_accepts_missing_statistics(graph_with_nodes: HeteroData) -> None:
    """NoResidualScaler is the no-op analogue of NoTendencyScaler: it never needs statistics."""
    del graph_with_nodes
    data_indices = _data_indices()

    scaler = NoResidualScaler(data_indices=data_indices, statistics=STATISTICS, statistics_residuals=None)
    scaling = scaler.get_scaling_values()

    assert torch.allclose(scaling, torch.ones_like(scaling))


# A prognostic variable sitting at data-output index 0. Under the old truthiness check
# (``name_to_index.get(key)`` returning a falsy 0) this variable was silently skipped and left at
# unit scaling; the ``is not None`` fix scales it correctly.
INDEX0_NAME_TO_INDEX = {"d": 0, "x": 1}
INDEX0_STATISTICS = {"stdev": [10.0, 1.0]}
INDEX0_STATISTICS_RESIDUALS = {"stdev": [2.0, 1.0]}


def _index0_config() -> DictConfig:
    return DictConfig(
        {
            "data": {"forcing": ["x"], "diagnostic": []},
            "training": {
                "variable_groups": {"default": "sfc"},
            },
        },
    )


def test_stdev_residual_scaler_scales_prognostic_at_data_output_index_0() -> None:
    """A prognostic variable at data-output index 0 must be scaled, not skipped (index-0 bug)."""
    config = _index0_config()
    data_indices = IndexCollection(data_config=config.data, name_to_index=INDEX0_NAME_TO_INDEX)

    scaler = StdevResidualScaler(
        data_indices=data_indices,
        statistics=INDEX0_STATISTICS,
        statistics_residuals=INDEX0_STATISTICS_RESIDUALS,
    )
    scaling = scaler.get_scaling_values()

    # "d" is prognostic at global/data-output index 0 -> stdev/residual_stdev = 10/2 = 5.0.
    # Under the old ``.get(key,)`` truthiness bug this stayed 1.0.
    d_output_pos = data_indices.data.output.positions_for_names(["d"])[0]
    assert scaling[d_output_pos].item() == pytest.approx(5.0)


@pytest.mark.parametrize("scaler_cls", [StdevResidualScaler, VarResidualScaler])
def test_residual_scalers_require_explicit_statistics(scaler_cls: type) -> None:
    """Stdev/Var residual scalers must fail hard rather than silently fall back to state stats."""
    data_indices = _data_indices()

    with pytest.raises(ValueError, match="statistics_residuals"):
        scaler_cls(data_indices=data_indices, statistics=STATISTICS, statistics_residuals=None)

    with pytest.raises(ValueError, match="statistics_residuals"):
        scaler_cls(
            data_indices=data_indices,
            statistics=STATISTICS,
            statistics_residuals={"mean": [0.0]},  # missing "stdev"
        )

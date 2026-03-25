# (C) Copyright 2025- Anemoi contributors.
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

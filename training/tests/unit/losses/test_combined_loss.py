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
from hydra.errors import InstantiationException
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper


def test_combined_loss() -> None:
    """Test the combined loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "scalers": ["test"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" in loss.losses[1].scaler


def test_combined_loss_invalid_loss_weights() -> None:
    """Test the combined loss function with invalid loss weights."""
    with pytest.raises(InstantiationException):
        get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.combined.CombinedLoss",
                    "losses": [
                        {"_target_": "anemoi.training.losses.MSELoss"},
                        {"_target_": "anemoi.training.losses.MAELoss"},
                    ],
                    "scalers": ["test"],
                    "loss_weights": [1.0, 0.5, 1],
                },
            ),
            scalers={"test": (-1, torch.ones(2))},
        )


def test_combined_loss_equal_weighting() -> None:
    """Test equal weighting when not given."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
            },
        ),
        scalers={},
    )
    assert all(weight == 1.0 for weight in loss.loss_weights)


def test_combined_loss_seperate_scalers() -> None:
    """Test that scalers are passed to the correct loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["test"]},
                    {"_target_": "anemoi.training.losses.MAELoss", "scalers": ["test2"]},
                ],
                "scalers": ["test", "test2"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2)), "test2": (-1, torch.ones(2))},
    )
    assert isinstance(loss, CombinedLoss)

    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler
    assert "test2" not in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" not in loss.losses[1].scaler
    assert "test2" in loss.losses[1].scaler


def test_combined_loss_multiscale_graph_data() -> None:
    graph = HeteroData()
    graph["data"].num_nodes = 4
    graph["smooth_8x"].num_nodes = 4
    graph["smooth_4x"].num_nodes = 4
    graph["smooth_2x"].num_nodes = 4
    graph["smooth_1x"].num_nodes = 4
    identity_edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    for name in ["smooth_8x", "smooth_4x", "smooth_2x", "smooth_1x"]:
        graph[(name, "to", name)].edge_index = identity_edges
        graph[(name, "to", name)].edge_weight = torch.ones(4)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                        "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
                        "keep_batch_sharded": False,
                        "loss_matrices_graph": [
                            {"edges_name": ["smooth_8x", "to", "smooth_8x"], "edge_weight_attribute": "edge_weight"},
                            {"edges_name": ["smooth_4x", "to", "smooth_4x"], "edge_weight_attribute": "edge_weight"},
                            {"edges_name": ["smooth_2x", "to", "smooth_2x"], "edge_weight_attribute": "edge_weight"},
                            {"edges_name": ["smooth_1x", "to", "smooth_1x"], "edge_weight_attribute": "edge_weight"},
                            None,
                        ],
                        "per_scale_loss": {"_target_": "anemoi.training.losses.MSELoss"},
                    },
                ],
                "loss_weights": [1.0],
            },
        ),
        scalers={},
        graph_data=graph,
    )

    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.losses[0], MultiscaleLossWrapper)
    assert loss.losses[0].smoothing_matrices[-1] is None

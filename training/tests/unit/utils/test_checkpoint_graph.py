# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.training.utils.checkpoint import check_node_coordinates_match_checkpoint
from anemoi.training.utils.checkpoint import load_graph_from_checkpoint


class _NodeAttributes(nn.Module):
    def __init__(self, coords: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("latlons_hidden", torch.cat([coords.sin(), coords.cos()], dim=-1), persistent=True)
        self.linear = nn.Linear(2, 2)


def _state_dict_for(coords: torch.Tensor) -> dict:
    return _NodeAttributes(coords).state_dict()


def test_matching_node_coordinates_pass() -> None:
    coords = torch.rand(10, 2)
    check_node_coordinates_match_checkpoint(_NodeAttributes(coords), _state_dict_for(coords))


def test_reordered_node_coordinates_raise() -> None:
    coords = torch.rand(10, 2)
    model = _NodeAttributes(coords[torch.randperm(10)])

    with pytest.raises(RuntimeError, match="latlons_hidden: .*nodes differ"):
        check_node_coordinates_match_checkpoint(model, _state_dict_for(coords))


def test_different_number_of_nodes_raises() -> None:
    model = _NodeAttributes(torch.rand(8, 2))

    with pytest.raises(RuntimeError, match="latlons_hidden: shape"):
        check_node_coordinates_match_checkpoint(model, _state_dict_for(torch.rand(10, 2)))


def test_checkpoint_without_node_coordinates_is_skipped() -> None:
    check_node_coordinates_match_checkpoint(_NodeAttributes(torch.rand(10, 2)), {})


def test_load_graph_from_checkpoint(tmp_path: Path) -> None:
    graph = HeteroData()
    graph["hidden"].x = torch.rand(5, 2)
    ckpt_path = tmp_path / "last.ckpt"
    torch.save({"hyper_parameters": {"graph_data": graph}}, ckpt_path)

    loaded = load_graph_from_checkpoint(ckpt_path)

    assert torch.equal(loaded["hidden"].x, graph["hidden"].x)


def test_load_graph_from_checkpoint_without_graph_returns_none(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "last.ckpt"
    torch.save({"hyper_parameters": {"graph_data": {"era5": HeteroData()}}}, ckpt_path)

    assert load_graph_from_checkpoint(ckpt_path) is None

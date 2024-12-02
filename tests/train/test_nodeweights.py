# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from anemoi.graphs.nodes.attributes import AreaWeights
from torch_geometric.data import HeteroData

from anemoi.training.losses.nodeweights import GraphNodeAttribute
from anemoi.training.losses.nodeweights import ReweightedGraphNodeAttribute


def fake_graph() -> HeteroData:
    hdata = HeteroData()
    lons = torch.tensor([1.56, 3.12, 4.68, 6.24])
    lats = torch.tensor([-3.12, -1.56, 1.56, 3.12])
    cutout_mask = torch.tensor([False, True, False, False]).unsqueeze(1)
    area_weights = torch.ones(cutout_mask.shape)
    hdata["data"]["x"] = torch.stack((lats, lons), dim=1)
    hdata["data"]["cutout"] = cutout_mask
    hdata["data"]["area_weight"] = area_weights

    return hdata


def fake_sv_area_weights() -> torch.Tensor:
    return AreaWeights(norm="unit-max", fill_value=0).compute(fake_graph(), "data").squeeze()


def fake_reweighted_sv_area_weights(frac: float) -> torch.Tensor:
    weights = fake_sv_area_weights().unsqueeze(1)
    cutout_mask = fake_graph()["data"]["cutout"]
    unmasked_sum = torch.sum(weights[~cutout_mask])
    weight_per_masked_node = frac / (1.0 - frac) * unmasked_sum / sum(cutout_mask)
    weights[cutout_mask] = weight_per_masked_node

    return weights.squeeze()


@pytest.mark.parametrize(
    ("target_nodes", "node_attribute", "fake_graph", "expected_weights"),
    [
        ("data", "area_weight", fake_graph(), fake_graph()["data"]["area_weight"]),
        ("data", "non_existent_attr", fake_graph(), fake_sv_area_weights()),
    ],
)
def test_grap_node_attributes(
    target_nodes: str,
    node_attribute: str,
    fake_graph: HeteroData,
    expected_weights: torch.Tensor,
) -> None:
    weights = GraphNodeAttribute(target_nodes=target_nodes, node_attribute=node_attribute).weights(fake_graph)
    assert isinstance(weights, torch.Tensor)
    assert torch.allclose(weights, expected_weights)


@pytest.mark.parametrize(
    ("target_nodes", "node_attribute", "scaled_attribute", "weight_frac_of_total", "fake_graph", "expected_weights"),
    [
        ("data", "area_weight", "cutout", 0.0, fake_graph(), torch.tensor([1.0, 0.0, 1.0, 1.0])),
        ("data", "area_weight", "cutout", 0.5, fake_graph(), torch.tensor([1.0, 3.0, 1.0, 1.0])),
        ("data", "area_weight", "cutout", 0.97, fake_graph(), torch.tensor([1.0, 97.0, 1.0, 1.0])),
        ("data", "non_existent_attr", "cutout", 0.0, fake_graph(), fake_reweighted_sv_area_weights(0.0)),
        ("data", "non_existent_attr", "cutout", 0.5, fake_graph(), fake_reweighted_sv_area_weights(0.5)),
        ("data", "non_existent_attr", "cutout", 0.99, fake_graph(), fake_reweighted_sv_area_weights(0.99)),
    ],
)
def test_graph_node_attributes(
    target_nodes: str,
    node_attribute: str,
    scaled_attribute: str,
    weight_frac_of_total: float,
    fake_graph: HeteroData,
    expected_weights: torch.Tensor,
) -> None:
    weights = ReweightedGraphNodeAttribute(
        target_nodes=target_nodes,
        node_attribute=node_attribute,
        scaled_attribute=scaled_attribute,
        weight_frac_of_total=weight_frac_of_total,
    ).weights(graph_data=fake_graph)
    assert isinstance(weights, torch.Tensor)
    assert torch.allclose(weights, expected_weights)

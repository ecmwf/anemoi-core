# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

import anemoi.models.interface as interface_module
from anemoi.models.interface import AnemoiModelInterface


def _bare_interface(config, graph: HeteroData, target_anchors: dict[str, str] | None = None) -> AnemoiModelInterface:
    model_interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(model_interface)
    model_interface.config = config
    model_interface.graph_data = graph
    model_interface.statistics = {}
    model_interface.statistics_tendencies = None
    model_interface.residual_statistics = None
    model_interface.data_indices = {}
    model_interface.n_step_input = 1
    model_interface.n_step_output = 1
    model_interface.target_anchors = target_anchors
    return model_interface


def test_interface_passes_complete_graph_to_spatial_preprocessor() -> None:
    graph = HeteroData()
    graph["source"].num_nodes = 2
    graph["projected"].num_nodes = 1
    graph["source", "to", "projected"].edge_index = torch.tensor([[0, 1], [0, 0]])

    config = OmegaConf.create(
        {
            "data": {
                "datasets": {
                    "projected": {
                        "processors": {},
                        "spatial_processor": {
                            "_target_": "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector",
                            "edges_name": ["source", "to", "projected"],
                        },
                    }
                },
            },
            "model": {"model": {"_target_": "torch.nn.Identity"}},
        }
    )
    model_interface = _bare_interface(config, graph)

    model_interface._build_model()

    projector = model_interface.spatial_pre_processors["projected"]
    projected, grid_shard_sizes = projector(torch.tensor([[[[[1.0], [3.0]]]]]))

    assert grid_shard_sizes is None
    torch.testing.assert_close(projected, torch.tensor([[[[[2.0]]]]]))


@pytest.mark.parametrize("target_anchors", [None, {"out_hres": "in_lres"}])
def test_interface_forwards_target_anchors_to_the_model_only_when_set(
    monkeypatch: pytest.MonkeyPatch,
    target_anchors: dict[str, str] | None,
) -> None:
    """Models that know nothing about target anchors must not receive the argument."""
    seen_kwargs: dict = {}

    def _instantiate(config, **kwargs):
        if config.get("_target_") == "torch.nn.Identity":
            seen_kwargs.update(kwargs)
            return torch.nn.Identity()
        raise AssertionError(f"Unexpected instantiation of {config}")

    monkeypatch.setattr(interface_module, "instantiate", _instantiate)
    config = OmegaConf.create({"data": {"datasets": {}}, "model": {"model": {"_target_": "torch.nn.Identity"}}})
    model_interface = _bare_interface(config, HeteroData(), target_anchors=target_anchors)

    model_interface._build_model()

    if target_anchors is None:
        assert "target_anchors" not in seen_kwargs
    else:
        assert seen_kwargs["target_anchors"] == target_anchors

# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Constructing model building-blocks from placeholders, then filling them via state_dict.

This validates the construct-without-graph / construct-without-statistics path: build each
module from a zero-filled placeholder (so shapes match), then ``load_state_dict`` the real
values and assert equality. The placeholders are derived only from the JSON reconstruction
bundle + data_indices — no pickled graph / statistics objects.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.checkpoint import build_placeholder_graph
from anemoi.models.checkpoint import build_placeholder_statistics
from anemoi.models.checkpoint import serialise_graph_structure
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph_provider import StaticGraphProvider
from anemoi.models.preprocessing.normalizer import InputNormalizer


def _real_graph(seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    graph = HeteroData()
    graph["data"].x = torch.rand(12, 2, generator=generator)  # lat/lon coords
    graph["data"].num_nodes = 12
    graph["hidden"].x = torch.rand(5, 2, generator=generator)
    graph["hidden"].num_nodes = 5
    edge_store = graph[("data", "to", "hidden")]
    edge_store.edge_index = torch.randint(0, 5, (2, 20), generator=generator)
    edge_store["edge_length"] = torch.rand(20, 1, generator=generator)
    edge_store["edge_dirs"] = torch.rand(20, 2, generator=generator)
    return graph


def test_placeholder_graph_matches_shapes():
    real = _real_graph()
    summary = serialise_graph_structure(real)
    placeholder = build_placeholder_graph(summary)

    assert placeholder["data"].num_nodes == 12
    assert placeholder["hidden"].num_nodes == 5
    assert placeholder["data"].x.shape == real["data"].x.shape
    edge = placeholder[("data", "to", "hidden")]
    assert edge.edge_index.shape == real[("data", "to", "hidden")].edge_index.shape
    assert edge["edge_length"].shape[-1] == 1
    assert edge["edge_dirs"].shape[-1] == 2


def test_static_graph_provider_reconstructs_from_placeholder():
    real = _real_graph(seed=1)
    edge_type = ("data", "to", "hidden")
    summary = serialise_graph_structure(real)
    placeholder = build_placeholder_graph(summary)

    kwargs = dict(edge_attributes=["edge_length", "edge_dirs"], src_size=12, dst_size=5, trainable_size=0)
    real_provider = StaticGraphProvider(graph=real[edge_type], **kwargs)
    ph_provider = StaticGraphProvider(graph=placeholder[edge_type], **kwargs)

    # Same shapes / edge_dim built from the placeholder.
    assert ph_provider.edge_dim == real_provider.edge_dim
    assert ph_provider.edge_index_base.shape == real_provider.edge_index_base.shape
    assert ph_provider.edge_attr.shape == real_provider.edge_attr.shape

    # Filling the placeholder provider from the real state_dict restores the graph.
    ph_provider.load_state_dict(real_provider.state_dict(), strict=True)
    assert torch.equal(ph_provider.edge_index_base, real_provider.edge_index_base)
    assert torch.equal(ph_provider.edge_attr, real_provider.edge_attr)


def test_named_nodes_attributes_reconstructs_from_placeholder():
    real = _real_graph(seed=2)
    summary = serialise_graph_structure(real)
    placeholder = build_placeholder_graph(summary)

    trainable = {"data": 0, "hidden": 0}
    real_nna = NamedNodesAttributes(trainable, real)
    ph_nna = NamedNodesAttributes(trainable, placeholder)

    # attr_ndims (= 2 * coord_dim + trainable) sizes the encoders -> must match.
    assert ph_nna.attr_ndims == real_nna.attr_ndims
    assert ph_nna.num_nodes == real_nna.num_nodes

    ph_nna.load_state_dict(real_nna.state_dict(), strict=True)
    assert torch.equal(ph_nna.latlons_data, real_nna.latlons_data)
    assert torch.equal(ph_nna.latlons_hidden, real_nna.latlons_hidden)


def _normalizer_setup():
    config = DictConfig(
        {
            "data": {
                "normalizer": {"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"]},
                "forcing": ["z"],
                "diagnostic": ["other"],
            },
        },
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    data_indices = IndexCollection(data_config=config.data, name_to_index=name_to_index)
    return config, data_indices


def test_input_normalizer_reconstructs_from_placeholder_statistics():
    config, data_indices = _normalizer_setup()
    real_statistics = {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1.0, 14.0]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }
    real = InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=real_statistics)

    # Placeholder statistics (sized from data_indices) build a valid but neutral normalizer.
    placeholder_statistics = build_placeholder_statistics({"data": data_indices})["data"]
    ph = InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=placeholder_statistics)

    assert ph._norm_mul.shape == real._norm_mul.shape
    # Neutral placeholder differs from the real scale...
    assert not torch.allclose(ph._norm_mul, real._norm_mul)

    # ...until we load the real state_dict.
    ph.load_state_dict(real.state_dict(), strict=True)
    assert torch.allclose(ph._norm_mul, real._norm_mul)
    assert torch.allclose(ph._norm_add, real._norm_add)

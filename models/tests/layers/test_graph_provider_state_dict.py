# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The static graph provider must carry its edge tensors in the state_dict.

This is part of the no-pickle effort: the graph topology / edge attributes should travel
with the weights through ``state_dict`` instead of a separately-pickled graph object.
"""

import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import StaticGraphProvider


def _edge_store(num_edges: int = 5, attr_dim: int = 3, seed: int = 0):
    """Build a small edge store with `edge_index` and an `edge_length` attribute."""
    generator = torch.Generator().manual_seed(seed)
    graph = HeteroData()
    graph["src"].num_nodes = 6
    graph["dst"].num_nodes = 4
    edge_store = graph[("src", "to", "dst")]
    edge_store.edge_index = torch.randint(0, 4, (2, num_edges), generator=generator)
    edge_store["edge_length"] = torch.rand(num_edges, attr_dim, generator=generator)
    return edge_store


def _make_provider(seed: int = 0) -> StaticGraphProvider:
    return StaticGraphProvider(
        graph=_edge_store(seed=seed),
        edge_attributes=["edge_length"],
        src_size=6,
        dst_size=4,
        trainable_size=0,
    )


def test_edge_tensors_are_in_state_dict():
    provider = _make_provider()
    keys = set(provider.state_dict().keys())
    assert {"edge_attr", "edge_index_base", "edge_inc"} <= keys
    # `perm` is construction-only and must NOT be persisted.
    assert "perm" not in keys


def test_state_dict_round_trip_restores_graph():
    src = _make_provider(seed=1)
    state_dict = src.state_dict()

    # A provider built from a *different* graph...
    dst = _make_provider(seed=2)
    assert not torch.equal(dst.edge_index_base, src.edge_index_base)

    # ...takes on the saved graph after loading.
    dst.load_state_dict(state_dict, strict=True)
    assert torch.equal(dst.edge_index_base, src.edge_index_base)
    assert torch.equal(dst.edge_attr, src.edge_attr)
    assert torch.equal(dst.edge_inc, src.edge_inc)


def test_old_checkpoint_without_edge_buffers_still_loads_strict():
    """Checkpoints written before edge buffers were persisted lack these keys.

    They must still load under strict=True because the buffers are rebuilt from the graph
    at construction time.
    """
    provider = _make_provider(seed=3)
    legacy_state_dict = provider.state_dict()
    for key in ("edge_attr", "edge_index_base", "edge_inc"):
        del legacy_state_dict[key]

    fresh = _make_provider(seed=3)
    # Must not raise despite strict=True and the missing keys.
    fresh.load_state_dict(legacy_state_dict, strict=True)
    # Buffers keep their construction-time values.
    assert fresh.edge_index_base.shape[0] == 2

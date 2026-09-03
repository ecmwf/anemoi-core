# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Serialise a model / graph to a JSON-able dict and rebuild it with the builder.

- A model built by ``build_model`` records its architecture recipe. ``to_dict(model)``
  returns that recipe as plain (JSON-dumpable) types; feeding it back to ``build_model``
  reconstructs an identical architecture (same ``state_dict`` keys, same parameter count).
- A graph is defined by its recipe (the graph config). It is already basic types, so it
  round-trips through JSON and ``GraphCreator`` gives an identical graph.

Run:  python examples/serialise_roundtrip.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_model import build_from_config  # noqa: E402
from create_model import make_data_indices  # noqa: E402
from create_model import make_graph  # noqa: E402

from anemoi.models.models.builder import build_model  # noqa: E402
from anemoi.utils.builder import to_dict  # noqa: E402


def roundtrip_model() -> None:
    print("== model ==")
    model = build_from_config()

    # Serialise -> JSON string -> back to a plain dict.
    spec = to_dict(model)
    json_text = json.dumps(spec)
    print(f"  serialised recipe: {len(json_text)} bytes of JSON")
    spec_back = json.loads(json_text)

    # Rebuild from the serialised recipe (runtime context supplied again).
    rebuilt = build_model(
        spec_back,
        data_indices=make_data_indices(),
        statistics={"data": None},
        graph_data=make_graph(),
        n_step_input=1,
        n_step_output=1,
    )

    same_keys = set(model.state_dict()) == set(rebuilt.state_dict())
    same_params = sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in rebuilt.parameters())
    print(f"  same state_dict keys : {same_keys}")
    print(f"  same parameter count : {same_params} ({sum(p.numel() for p in model.parameters())})")
    assert same_keys and same_params, "model did not round-trip"
    print("  model round-trip OK")


def _graph_config(npz_path: str) -> dict:
    return {
        "nodes": {
            "test_nodes": {"node_builder": {"_target_": "anemoi.graphs.nodes.NPZFileNodes", "npz_file": npz_path}}
        },
        "edges": [
            {
                "source_name": "test_nodes",
                "target_name": "test_nodes",
                "edge_builders": [{"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}],
                "attributes": {
                    "dist_norm": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength"},
                    "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection"},
                },
            },
        ],
    }


def roundtrip_graph() -> None:
    print("== graph ==")
    from anemoi.graphs.create import GraphCreator

    tmp = tempfile.mkdtemp()
    npz = str(Path(tmp) / "grid.npz")
    np.savez(npz, latitudes=np.random.rand(40) * 180 - 90, longitudes=np.random.rand(40) * 360 - 180)

    config = _graph_config(npz)
    # The recipe is already basic types -> JSON round-trips trivially.
    config_back = json.loads(json.dumps(config))
    assert config_back == config

    graph_a = GraphCreator(config=config).create()
    graph_b = GraphCreator(config=config_back).create()

    same_nodes = graph_a.node_types == graph_b.node_types
    same_edges = graph_a.edge_types == graph_b.edge_types
    edge = ("test_nodes", "to", "test_nodes")
    same_attrs = set(graph_a[edge].edge_attrs()) == set(graph_b[edge].edge_attrs())
    same_shape = graph_a[edge].edge_index.shape == graph_b[edge].edge_index.shape
    print(f"  node types match     : {same_nodes} ({graph_a.node_types})")
    print(f"  edge attrs match     : {same_attrs} ({sorted(graph_a[edge].edge_attrs())})")
    print(f"  edge_index shape     : {same_shape} ({tuple(graph_a[edge].edge_index.shape)})")
    assert same_nodes and same_edges and same_attrs and same_shape, "graph did not round-trip"
    print("  graph round-trip OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    roundtrip_model()
    roundtrip_graph()
    print("ALL OK")

# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.checkpoint import RECONSTRUCTION_KEY
from anemoi.models.checkpoint import add_reconstruction_metadata
from anemoi.models.checkpoint import build_reconstruction_metadata
from anemoi.models.checkpoint import load_reconstruction_metadata
from anemoi.models.checkpoint import rebuild_data_indices
from anemoi.models.checkpoint import serialise_graph_structure
from anemoi.models.data_indices.collection import IndexCollection


def _make_index_collection() -> IndexCollection:
    data_config = OmegaConf.create({"forcing": ["lsm", "z"], "diagnostic": ["tp"]})
    name_to_index = {"2t": 0, "10u": 1, "lsm": 2, "z": 3, "tp": 4}
    return IndexCollection(data_config, name_to_index)


def test_index_collection_round_trip():
    original = _make_index_collection()
    rebuilt = IndexCollection.from_serialised(original.to_serialised())
    # IndexCollection.__eq__ compares the derived data + model index spaces.
    assert rebuilt == original
    assert rebuilt.name_to_index == original.name_to_index
    assert torch.equal(rebuilt.data.input.full, original.data.input.full)
    assert torch.equal(rebuilt.model.output.full, original.model.output.full)


def test_to_serialised_is_json_safe():
    payload = _make_index_collection().to_serialised()
    # Must survive a JSON round-trip unchanged.
    assert json.loads(json.dumps(payload)) == payload


def test_build_reconstruction_metadata():
    data_indices = {"dataset": _make_index_collection()}
    bundle = build_reconstruction_metadata(
        data_indices=data_indices,
        n_step_input=2,
        n_step_output=1,
    )
    assert bundle["n_step_input"] == 2
    assert bundle["n_step_output"] == 1
    assert "dataset" in bundle["data_indices"]
    rebuilt = rebuild_data_indices(bundle)
    assert rebuilt["dataset"] == data_indices["dataset"]


def test_serialise_graph_structure():
    graph = HeteroData()
    graph["data"].x = torch.zeros(40, 2)
    graph["data"].num_nodes = 40
    graph["hidden"].x = torch.zeros(10, 2)
    graph["hidden"].num_nodes = 10
    edge_store = graph[("data", "to", "hidden")]
    edge_store.edge_index = torch.randint(0, 10, (2, 25))
    edge_store["edge_length"] = torch.rand(25, 1)

    summary = serialise_graph_structure(graph)
    assert summary["nodes"]["data"]["num_nodes"] == 40
    assert summary["nodes"]["hidden"]["num_nodes"] == 10
    assert summary["edges"]["data,to,hidden"]["num_edges"] == 25
    assert summary["edges"]["data,to,hidden"]["attributes"]["edge_length"] == 1
    # Must be JSON-safe.
    assert json.loads(json.dumps(summary)) == summary


def test_add_and_load_reconstruction_metadata_in_checkpoint(tmp_path):
    """Write the bundle into a real torch checkpoint file via anemoi.utils, read it back."""
    bundle = build_reconstruction_metadata(
        data_indices={"dataset": _make_index_collection()},
        n_step_input=2,
        n_step_output=1,
    )

    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(3)}}, ckpt_path)

    # No reconstruction metadata yet.
    assert load_reconstruction_metadata(str(ckpt_path)) is None

    add_reconstruction_metadata(str(ckpt_path), bundle)

    loaded = load_reconstruction_metadata(str(ckpt_path))
    assert loaded is not None
    assert loaded["n_step_input"] == 2
    rebuilt = rebuild_data_indices(loaded)
    assert rebuilt["dataset"] == _make_index_collection()


def test_add_reconstruction_metadata_preserves_existing(tmp_path):
    """Adding the bundle must not clobber existing Anemoi metadata (e.g. config)."""
    from anemoi.utils.checkpoints import load_metadata
    from anemoi.utils.checkpoints import save_metadata

    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)
    save_metadata(str(ckpt_path), {"version": "1.0.0", "config": {"model": {"num_channels": 8}}})

    bundle = build_reconstruction_metadata(
        data_indices={"dataset": _make_index_collection()},
        n_step_input=1,
        n_step_output=1,
    )
    add_reconstruction_metadata(str(ckpt_path), bundle)

    metadata = load_metadata(str(ckpt_path))
    assert metadata["config"]["model"]["num_channels"] == 8  # preserved
    assert RECONSTRUCTION_KEY in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

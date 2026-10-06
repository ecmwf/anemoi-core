# (C) Copyright 2026 Anemoi contributors.

"""Query graph loading follows the shared graph utilities on current main."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs import utils
from anemoi.training.query import datamodule


def test_query_graph_loader_uses_shared_utilities(tmp_path, monkeypatch):
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")
    graph = HeteroData()
    graph["IFS"].x = torch.zeros(3, 2)
    graph["hidden"].x = torch.ones(2, 2)
    graph.query_geometry = {"datasets": ["IFS"]}
    filename = tmp_path / "query_graph.pt"
    torch.save(graph, filename)

    assert datamodule.load_graph_from_file is utils.load_graph_from_file
    assert datamodule.validate_loaded_graph is utils.validate_loaded_graph
    loaded = datamodule.load_graph_from_file(filename)
    datamodule.validate_loaded_graph(loaded, ["IFS", "hidden"])
    assert torch.equal(loaded["IFS"].x, graph["IFS"].x)
    assert loaded.query_geometry == graph.query_geometry
    with pytest.raises(ValueError, match="MEPS"):
        datamodule.validate_loaded_graph(loaded, ["IFS", "MEPS", "hidden"])

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.utils import FORCE_CPU_ENV_VAR

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_inspect_on_gpu(
    config_file: tuple[Path, str],
    mock_grids_path: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make sure that the inspect command runs when a GPU is available.

    The graph is created on the GPU. The inspect command must load the graph on the CPU to make the plots.
    """
    monkeypatch.delenv(FORCE_CPU_ENV_VAR, raising=False)
    tmp_path, config_name = config_file
    graph_path = tmp_path / "graph.pt"
    output_path = tmp_path / "inspect"

    GraphCreator(config=tmp_path / config_name).create(save_path=graph_path)
    assert graph_path.exists()

    result = subprocess.run(
        [sys.executable, "-m", "anemoi.graphs", "inspect", str(graph_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"anemoi-graphs inspect failed:\n{result.stdout}\n{result.stderr}"

    # The graph has no isolated nodes and no node attributes.
    # Thus the command does not create "isolated_nodes.html" and "distribution_node_attributes.png".
    expected_files = [
        "graph.html",
        "test_nodes_to_test_nodes.html",
        "distribution_node_adjancency.png",
        "distribution_edge_attributes.png",
    ]
    for file_name in expected_files:
        assert (output_path / file_name).exists(), f"{file_name} was not created."

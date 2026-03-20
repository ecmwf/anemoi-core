# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.training.utils.supporting_arrays import build_combined_supporting_arrays


def test_build_combined_supporting_arrays_uses_dataset_nodes_only() -> None:
    graph = HeteroData()
    graph["data"]["output_nodes"] = torch.tensor([True, False, True])
    graph["hidden"].x = torch.randn(2, 1)

    config = OmegaConf.create(
        {
            "data": {"processors": {}},
            "model": {
                "output_mask": {
                    "_target_": "anemoi.training.utils.masks.Boolean1DMask",
                    "attribute_name": "output_nodes",
                },
            },
        },
    )
    supporting_arrays = {"data": {"orography": [1, 2, 3]}}

    combined = build_combined_supporting_arrays(config, graph, supporting_arrays)

    assert "output_mask" in combined["data"]
    assert combined["data"]["output_mask"].tolist() == [True, False, True]
    assert "hidden" not in combined
    assert supporting_arrays == {"data": {"orography": [1, 2, 3]}}

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import OmegaConf

from anemoi.training.utils.graph_config import expand_projections_into_graph_config


def test_merge_projection_rewrites_local_node_builder_references_for_fused_graphs() -> None:
    graph_config = OmegaConf.create(
        {
            "nodes": {
                "era5": {},
                "cerra": {},
                "hidden": {},
            },
            "edges": [],
            "projections": {
                "multiscale": {
                    "smoothers": {
                        "smooth": {
                            "node_builder": {
                                "_target_": "anemoi.graphs.nodes.ReferenceNodes",
                                "reference_node_name": "data",
                            },
                            "num_nearest_neighbours": 8,
                        },
                        "smooth_ref": {
                            "node_builder": {
                                "_target_": "anemoi.graphs.nodes.ReferenceNodes",
                                "reference_node_name": "smooth",
                            },
                            "num_nearest_neighbours": 8,
                        },
                    },
                },
            },
        },
    )

    expand_projections_into_graph_config(graph_config, dataset_names=["era5", "cerra"])

    assert graph_config.nodes.era5_smooth.node_builder.reference_node_name == "era5"
    assert graph_config.nodes.cerra_smooth.node_builder.reference_node_name == "cerra"
    assert graph_config.nodes.era5_smooth_ref.node_builder.reference_node_name == "era5_smooth"
    assert graph_config.nodes.cerra_smooth_ref.node_builder.reference_node_name == "cerra_smooth"

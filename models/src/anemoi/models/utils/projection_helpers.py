# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Re-export shim — projection helpers have moved to anemoi.graphs."""

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_RELATION_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.projection_helpers import dataset_projection_node_name
from anemoi.graphs.projection_helpers import expand_geometric_smoothers
from anemoi.graphs.projection_helpers import get_graph_node_names
from anemoi.graphs.projection_helpers import multiscale_loss_matrices_graph
from anemoi.graphs.projection_helpers import projection_edge_name
from anemoi.graphs.projection_helpers import projection_node_name
from anemoi.graphs.projection_helpers import residual_projection_edge_names
from anemoi.graphs.projection_helpers import residual_projection_truncation_node_name
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph

__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_EDGE_RELATION_NAME",
    "DEFAULT_EDGE_WEIGHT_ATTRIBUTE",
    "dataset_projection_node_name",
    "expand_geometric_smoothers",
    "get_graph_node_names",
    "multiscale_loss_matrices_graph",
    "projection_edge_name",
    "projection_node_name",
    "residual_projection_edge_names",
    "residual_projection_truncation_node_name",
    "uses_fused_dataset_graph",
]

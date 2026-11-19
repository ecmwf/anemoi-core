# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import unittest

import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.attributes import NodeFeatures


class TestNodeFeatures(unittest.TestCase):
    def test_endpoint_values_follow_edge_indices(self):
        edges = torch.tensor([[0, 1, 0], [1, 0, 1]])
        nodes = NodeStorage(x=torch.zeros(2, 2), features=torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
        for endpoint, axis in (("source", 0), ("target", 1)):
            with self.subTest(endpoint=endpoint):
                builder = NodeFeatures(node_axis=endpoint, num_features=2, dtype="float64")
                actual = builder(x=(nodes, nodes), edge_index=edges)
                expected = nodes.features[edges[axis]].to(device=actual.device, dtype=torch.float64)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(builder.ndim, 2)

    def test_invalid_configuration(self):
        for kwargs in ({"node_axis": "other"}, {"num_features": 0}, {"dtype": "int64"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NodeFeatures(**({"node_axis": "target", "num_features": 1} | kwargs))

    def test_missing_or_misaligned_features(self):
        for features in (None, torch.ones(2, 2), torch.ones(3, 1), torch.ones(2, 1, 1)):
            nodes = NodeStorage(x=torch.zeros(2, 2))
            if features is not None:
                nodes.features = features
            with self.subTest(features=features), self.assertRaises(ValueError):
                NodeFeatures(node_axis="target", num_features=1)(x=(nodes, nodes), edge_index=torch.tensor([[0], [0]]))


if __name__ == "__main__":
    unittest.main()

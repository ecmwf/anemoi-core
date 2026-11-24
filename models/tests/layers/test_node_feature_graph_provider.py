# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import unittest

import torch

from anemoi.graphs.edges.attributes import AttributeFromNodeToEdge
from anemoi.models.layers.graph_provider import DynamicGraphProvider


class _AllEdges:
    @staticmethod
    def compute_edge_index_from_coords(source_coords, target_coords):
        # Deliberately not sorted by destination, to check attribute permutation too.
        source = torch.arange(len(source_coords), device=source_coords.device).repeat_interleave(len(target_coords))
        target = torch.arange(len(target_coords), device=target_coords.device).repeat(len(source_coords))
        return torch.stack((source, target))


def _provider(endpoint="target"):
    provider = DynamicGraphProvider.__new__(DynamicGraphProvider)
    torch.nn.Module.__init__(provider)
    provider.edge_builder = _AllEdges()
    provider.attributes_config = {"level": AttributeFromNodeToEdge(node_axis=endpoint, num_features=1)}
    provider._edge_dim = 1
    provider._capture_request = None
    provider._captured_graph = None
    return provider


class TestNodeFeatureGraphProvider(unittest.TestCase):
    def test_batched_features_preserve_sample_boundaries_and_empty_samples(self):
        provider = _provider()
        features = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
        attrs, edges = provider.build_graph(
            torch.zeros(3, 2),
            torch.zeros(5, 2),
            src_batch_sizes=(1, 0, 2),
            dst_batch_sizes=(2, 1, 2),
            dst_node_features=features,
        )
        torch.testing.assert_close(attrs, features[edges[1]])
        self.assertEqual(attrs.shape, (6, 1))
        self.assertFalse(torch.any(edges[1] == 2).item())
        self.assertTrue(torch.all(edges[0, edges[1] < 2] == 0).item())
        self.assertTrue(torch.all(edges[0, edges[1] >= 3] >= 1).item())

    def test_checkpoint_capture_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for endpoint, axis in (("source", 0), ("target", 1)):
                for checkpoint in (False, True):
                    with self.subTest(device=device, endpoint=endpoint, checkpoint=checkpoint):
                        provider = _provider(endpoint)
                        provider.capture_next_graph("hidden", "obs")
                        features = torch.tensor([[10.0], [20.0]], device=device, requires_grad=True)
                        attrs, edges, _ = provider.get_edges(
                            src_coords=torch.zeros(2, 2, device=device),
                            dst_coords=torch.zeros(2, 2, device=device),
                            **{f"{'src' if axis == 0 else 'dst'}_node_features": features},
                            shard_edges=False,
                            act_checkpoint=checkpoint,
                        )
                        torch.testing.assert_close(attrs, features[edges[axis]])
                        graph = provider.consume_captured_graph()
                        torch.testing.assert_close(graph["hidden", "to", "obs"].level, attrs.cpu())
                        attrs.sum().backward()
                        torch.testing.assert_close(features.grad, torch.full_like(features, 2.0))

    def test_invalid_feature_rows(self):
        for features in (torch.ones(3, 1), torch.ones(2), torch.tensor(1.0)):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, "one row per coordinate"):
                _provider().build_graph(torch.zeros(2, 2), torch.zeros(2, 2), dst_node_features=features)

    def test_empty_graph(self):
        attrs, edges, _ = _provider().get_edges(
            src_coords=torch.zeros(2, 2),
            dst_coords=torch.zeros(0, 2),
            dst_node_features=torch.empty(0, 1),
            shard_edges=False,
        )
        self.assertEqual(attrs.shape, (0, 1))
        self.assertEqual(edges.shape, (2, 0))


if __name__ == "__main__":
    unittest.main()

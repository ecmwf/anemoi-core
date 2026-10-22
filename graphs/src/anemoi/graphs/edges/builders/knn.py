"""Backward-compatible imports for checkpoints using the former KNN module path."""

from anemoi.graphs.edges.builders.euclidean.knn import (
    BaseKNNEdges,
    KNNEdges,
    MutualKNNEdges,
    ReversedKNNEdges,
)

__all__ = ["BaseKNNEdges", "KNNEdges", "MutualKNNEdges", "ReversedKNNEdges"]

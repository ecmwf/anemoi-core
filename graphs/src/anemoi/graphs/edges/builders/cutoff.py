"""Backward-compatible imports for checkpoints using the former cutoff module path."""

from anemoi.graphs.edges.builders.euclidean.cutoff import (
    BaseCutOffEdges,
    CutOffEdges,
    ReversedCutOffEdges,
)

__all__ = ["BaseCutOffEdges", "CutOffEdges", "ReversedCutOffEdges"]

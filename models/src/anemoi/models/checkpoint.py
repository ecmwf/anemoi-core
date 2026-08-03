# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pickle-free reconstruction metadata for Anemoi checkpoints.

Once the model is built from the resolved ``config`` (native ``instantiate`` backend) and
the graph / data-index *tensors* travel in the ``state_dict`` (registered buffers), the
only remaining inputs needed to reconstruct a model are a handful of **non-tensor** facts:

* the ``name_to_index`` mapping (and data config) for each dataset, from which the full
  ``IndexCollection`` is rebuilt;
* the number of input / output timesteps;
* a structural summary of the graph (node counts, edge counts, attribute dims) so the model
  can be constructed without the pickled graph object.

This module serialises those facts to a JSON-safe dict and stores it inside the checkpoint
file's Anemoi metadata using ``anemoi.utils.checkpoints`` (no pickling). See the no-pickle
docs at the repository root.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import numpy as np

from anemoi.models.data_indices.collection import IndexCollection

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)

# Key under which the reconstruction bundle is stored inside the Anemoi metadata dict.
RECONSTRUCTION_KEY = "reconstruction"

# Schema version of the bundle, so future readers can detect/upgrade older layouts.
RECONSTRUCTION_VERSION = 1


# --------------------------------------------------------------------------------------
# data_indices  <->  JSON
# --------------------------------------------------------------------------------------


def serialise_data_indices(data_indices: dict[str, IndexCollection]) -> dict[str, dict]:
    """Serialise a ``{dataset_name: IndexCollection}`` mapping to a JSON-safe dict."""
    return {name: collection.to_serialised() for name, collection in data_indices.items()}


def deserialise_data_indices(serialised: dict[str, dict]) -> dict[str, IndexCollection]:
    """Rebuild a ``{dataset_name: IndexCollection}`` mapping from :func:`serialise_data_indices`."""
    return {name: IndexCollection.from_serialised(payload) for name, payload in serialised.items()}


# --------------------------------------------------------------------------------------
# graph structural summary (tensors live in the state_dict, not here)
# --------------------------------------------------------------------------------------


def serialise_graph_structure(graph_data: "HeteroData") -> dict[str, Any]:
    """Summarise the graph topology in a JSON-safe way (no tensor data).

    The actual graph tensors travel in the model ``state_dict`` (registered buffers); this
    records only the *shape* of the graph (node counts, edge counts, attribute dims) so a
    model can be constructed without the pickled ``HeteroData`` object.
    """
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    try:
        node_types, edge_types = graph_data.metadata()
    except Exception:  # noqa: BLE001 - be defensive about non-standard graph objects
        node_types = list(getattr(graph_data, "node_types", []) or [])
        edge_types = list(getattr(graph_data, "edge_types", []) or [])

    for node_type in node_types:
        store = graph_data[node_type]
        num_nodes = getattr(store, "num_nodes", None)
        node_entry: dict[str, Any] = {"num_nodes": int(num_nodes) if num_nodes is not None else None}
        attrs = {}
        for key, value in store.items():
            if hasattr(value, "shape") and value.ndim >= 2:
                attrs[key] = int(value.shape[-1])
        if attrs:
            node_entry["attributes"] = attrs
        nodes[node_type] = node_entry

    for edge_type in edge_types:
        store = graph_data[edge_type]
        edge_entry: dict[str, Any] = {}
        edge_index = getattr(store, "edge_index", None)
        if edge_index is not None:
            edge_entry["num_edges"] = int(edge_index.shape[1])
        attrs = {}
        for key, value in store.items():
            if key == "edge_index":
                continue
            if hasattr(value, "shape") and value.ndim >= 2:
                attrs[key] = int(value.shape[-1])
        if attrs:
            edge_entry["attributes"] = attrs
        # JSON keys must be strings; an edge type is a (src, rel, dst) tuple.
        edges[_edge_type_key(edge_type)] = edge_entry

    return {"nodes": nodes, "edges": edges}


def _edge_type_key(edge_type: Any) -> str:
    if isinstance(edge_type, (tuple, list)):
        return ",".join(str(part) for part in edge_type)
    return str(edge_type)


# --------------------------------------------------------------------------------------
# the reconstruction bundle
# --------------------------------------------------------------------------------------


def build_reconstruction_metadata(
    *,
    data_indices: dict[str, IndexCollection],
    n_step_input: int,
    n_step_output: int,
    graph_data: Optional["HeteroData"] = None,
) -> dict[str, Any]:
    """Assemble the JSON-safe bundle of everything needed to rebuild a model pickle-free.

    Parameters
    ----------
    data_indices : dict[str, IndexCollection]
        Per-dataset index collections.
    n_step_input, n_step_output : int
        Number of input / output timesteps.
    graph_data : HeteroData, optional
        Graph object; only its structural summary is stored (no tensor data).
    """
    bundle: dict[str, Any] = {
        "version": RECONSTRUCTION_VERSION,
        "data_indices": serialise_data_indices(data_indices),
        "n_step_input": int(n_step_input),
        "n_step_output": int(n_step_output),
    }
    if graph_data is not None:
        bundle["graph"] = serialise_graph_structure(graph_data)
    return bundle


# --------------------------------------------------------------------------------------
# checkpoint I/O (via anemoi.utils.checkpoints -- no pickling)
# --------------------------------------------------------------------------------------


def add_reconstruction_metadata(checkpoint_path: str, reconstruction: dict[str, Any]) -> None:
    """Store the reconstruction bundle inside the checkpoint's Anemoi metadata.

    Merges into the existing metadata (preserving ``config`` etc.) when present, otherwise
    creates a fresh metadata record. Uses ``anemoi.utils.checkpoints`` so nothing is pickled.
    """
    from anemoi.utils.checkpoints import has_metadata
    from anemoi.utils.checkpoints import load_metadata
    from anemoi.utils.checkpoints import replace_metadata
    from anemoi.utils.checkpoints import save_metadata

    if has_metadata(checkpoint_path):
        metadata = load_metadata(checkpoint_path)
        metadata[RECONSTRUCTION_KEY] = reconstruction
        # anemoi.utils.replace_metadata requires a top-level "version" key; real Anemoi
        # metadata always has one, but guard in case of a minimal/legacy record.
        metadata.setdefault("version", "1.0.0")
        replace_metadata(checkpoint_path, metadata)
    else:
        save_metadata(checkpoint_path, {"version": "1.0.0", RECONSTRUCTION_KEY: reconstruction})


def load_reconstruction_metadata(checkpoint_path: str) -> Optional[dict[str, Any]]:
    """Return the reconstruction bundle stored in the checkpoint, or ``None`` if absent."""
    from anemoi.utils.checkpoints import has_metadata
    from anemoi.utils.checkpoints import load_metadata

    if not has_metadata(checkpoint_path):
        return None
    metadata = load_metadata(checkpoint_path)
    return metadata.get(RECONSTRUCTION_KEY)


def rebuild_data_indices(reconstruction: dict[str, Any]) -> dict[str, IndexCollection]:
    """Convenience: rebuild the per-dataset ``IndexCollection`` map from a bundle."""
    return deserialise_data_indices(reconstruction["data_indices"])


def build_model_inputs(checkpoint_path: str) -> dict[str, Any]:
    """Assemble every ``AnemoiModelInterface`` constructor input (except weights), pickle-free.

    Reads the checkpoint's JSON metadata only (no ``torch.load``/unpickling): the resolved
    ``config`` plus the reconstruction bundle, from which ``data_indices`` is rebuilt and
    placeholder ``graph_data`` / ``statistics`` are sized. Construct the model with these
    (under the native ``instantiate`` backend) and then ``load_state_dict`` the weights to
    fill in the placeholder buffers.
    """
    from anemoi.utils.checkpoints import load_metadata

    metadata = load_metadata(checkpoint_path)
    if RECONSTRUCTION_KEY not in metadata:
        msg = (
            f"Checkpoint {checkpoint_path!r} has no '{RECONSTRUCTION_KEY}' metadata. It was "
            "written before pickle-free reconstruction was supported."
        )
        raise KeyError(msg)

    reconstruction = metadata[RECONSTRUCTION_KEY]
    data_indices = rebuild_data_indices(reconstruction)
    inputs: dict[str, Any] = {
        "config": metadata["config"],
        "data_indices": data_indices,
        "statistics": build_placeholder_statistics(data_indices),
        "n_step_input": reconstruction["n_step_input"],
        "n_step_output": reconstruction["n_step_output"],
        "metadata": metadata,
    }
    if "graph" in reconstruction:
        inputs["graph_data"] = build_placeholder_graph(reconstruction["graph"])
    return inputs


# --------------------------------------------------------------------------------------
# placeholders for constructing a model without the pickled graph / statistics
# --------------------------------------------------------------------------------------
#
# The graph tensors and the statistics-derived tensors (graph edge buffers, node coords,
# normalizer scale/offset) are all persistent buffers that travel in the state_dict. To
# construct a model without the pickled graph / statistics objects, build correctly-SHAPED
# placeholders so every module sizes its buffers/parameters identically; `load_state_dict`
# then overwrites the placeholder VALUES with the real ones from the checkpoint.


def build_placeholder_graph(graph_summary: dict[str, Any]) -> "HeteroData":
    """Build a zero-filled ``HeteroData`` matching the shapes in a graph summary.

    The result has the correct node counts, node-attribute dims (notably ``x``, which sizes
    the encoders), edge counts and edge-attribute dims, so a model builds with the right
    buffer/parameter shapes. The tensor *values* are placeholders, overwritten on
    ``load_state_dict``.
    """
    import torch
    from torch_geometric.data import HeteroData

    graph = HeteroData()
    for name, info in graph_summary.get("nodes", {}).items():
        num_nodes = info.get("num_nodes")
        graph[name].num_nodes = num_nodes
        for attr, dim in info.get("attributes", {}).items():
            graph[name][attr] = torch.zeros(num_nodes, dim)

    for key, info in graph_summary.get("edges", {}).items():
        edge_type = tuple(key.split(","))
        num_edges = info.get("num_edges", 0)
        store = graph[edge_type]
        store.edge_index = torch.zeros(2, num_edges, dtype=torch.long)
        for attr, dim in info.get("attributes", {}).items():
            store[attr] = torch.zeros(num_edges, dim)

    return graph


def build_placeholder_statistics(data_indices: dict[str, IndexCollection]) -> dict[str, dict]:
    """Build per-dataset placeholder statistics sized from ``data_indices``.

    Values are neutral (mean=0, stdev=1, min=0, max=1) so the normalizer/bounding build
    valid scale/offset buffers without dividing by zero; ``load_state_dict`` then restores
    the real ones. Sized to the number of dataset variables (``len(name_to_index)``).
    """
    statistics: dict[str, dict] = {}
    for name, collection in data_indices.items():
        n = len(collection.name_to_index)
        statistics[name] = {
            "minimum": np.zeros(n, dtype=np.float32),
            "maximum": np.ones(n, dtype=np.float32),
            "mean": np.zeros(n, dtype=np.float32),
            "stdev": np.ones(n, dtype=np.float32),
            # aliases used by some consumers
            "min": np.zeros(n, dtype=np.float32),
            "max": np.ones(n, dtype=np.float32),
            "stdev_tend": np.ones(n, dtype=np.float32),
        }
    return statistics


def build_reconstruction_metadata_from_interface(model_interface: Any) -> dict[str, Any]:
    """Build the reconstruction bundle from a constructed ``AnemoiModelInterface``.

    The interface already holds every non-tensor input needed: ``data_indices``,
    ``n_step_input`` / ``n_step_output`` and ``graph_data``. Call this at checkpoint-save
    time and pass the result to :func:`add_reconstruction_metadata`.
    """
    return build_reconstruction_metadata(
        data_indices=model_interface.data_indices,
        n_step_input=model_interface.n_step_input,
        n_step_output=model_interface.n_step_output,
        graph_data=getattr(model_interface, "graph_data", None),
    )

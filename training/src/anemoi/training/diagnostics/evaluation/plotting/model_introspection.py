# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Model-introspection helpers for plotting callbacks.

Owns *all* logic for pulling plot-relevant information out of
:class:`pytorch_lightning.LightningModule` (data indices, metadata, graph
artifacts). Each :func:`extract_*_inputs` function returns a ``dict`` of
keyword arguments matching the ``plot_fn`` signature documented in
``docs/modules/diagnostics.rst`` ("Plot function contracts").

Callbacks call these functions and splat the returned dict into ``plot_fn``::

    inputs = extract_loss_inputs(pl_module, dataset_name, parameter_groups)
    fig = plot_fn(loss, **inputs, step_index=i, settings=settings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from anemoi.training.diagnostics.evaluation.plotting.graph import get_edge_trainable_modules
from anemoi.training.diagnostics.evaluation.plotting.graph import get_node_trainable_tensors

if TYPE_CHECKING:
    import pytorch_lightning as pl


__all__ = [
    "extract_graph_inputs",
    "extract_loss_inputs",
    "extract_spatial_inputs",
    "unwrap_anemoi_model",
]


def unwrap_anemoi_model(pl_module: "pl.LightningModule") -> Any:
    """Return the underlying anemoi model, unwrapping DDP / interface layers."""
    return pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model


def extract_graph_inputs(pl_module: "pl.LightningModule", dataset_name: str) -> dict[str, Any]:
    """Return kwargs for a ``GraphTrainableFeaturesPlot`` ``plot_fn``.

    Keys: ``dataset_name``, ``node_attributes``, ``node_trainable_tensors``,
    ``edge_trainable_modules``. ``edge_trainable_modules`` is empty for
    hierarchical models (they carry no trainable edge parameters).
    """
    model = unwrap_anemoi_model(pl_module)
    return {
        "dataset_name": dataset_name,
        "node_attributes": model.node_attributes,
        "node_trainable_tensors": get_node_trainable_tensors(model.node_attributes),
        "edge_trainable_modules": get_edge_trainable_modules(model, dataset_name),
    }


def extract_loss_inputs(
    pl_module: "pl.LightningModule",
    dataset_name: str,
    parameter_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return kwargs for a ``PlotLoss`` ``plot_fn``.

    Keys: ``parameter_names`` (sorted by output-index), ``parameter_groups``
    (user-supplied grouping config), ``metadata_variables`` (dataset
    ``variables_metadata`` block or ``None``).
    """
    name_to_index = pl_module.data_indices[dataset_name].model.output.name_to_index
    names = list(name_to_index.keys())
    positions = list(name_to_index.values())
    parameter_names = [names[i] for i in np.argsort(positions)]

    metadata = pl_module.model.metadata
    metadata_variables = metadata["dataset"].get("variables_metadata") if metadata is not None else None

    return {
        "parameter_names": parameter_names,
        "parameter_groups": parameter_groups or {},
        "metadata_variables": metadata_variables,
    }


def extract_spatial_inputs(
    pl_module: "pl.LightningModule",
    dataset_name: str,
    parameters: list[str],
) -> dict[str, Any]:
    """Return kwargs for a ``SpatialMapPlot`` ``plot_fn``.

    Keys: ``parameters`` — mapping ``output_index -> (name, is_diagnostic)``.
    Per-batch tensors (``x``, ``y_true``, ``y_pred``, ``auxiliary``,
    ``latlons``) are threaded in by the callback itself.
    """
    indices = pl_module.data_indices[dataset_name]
    input_data = indices.data.input.todict()
    index_to_name = {v: k for k, v in input_data["name_to_index"].items()}
    diagnostics = {index_to_name[int(i)] for i in input_data["diagnostic"]}
    output_name_to_index = indices.model.output.name_to_index
    return {
        "parameters": {output_name_to_index[name]: (name, name in diagnostics) for name in parameters},
    }

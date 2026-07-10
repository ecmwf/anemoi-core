# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import logging
from collections.abc import Iterable
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from torch import Tensor

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.training.diagnostics.evaluation.geospatial.maps import map_features
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import _hide_axes_ticks

LOGGER = logging.getLogger(__name__)


class GraphPlotFn(Protocol):
    """Typing Protocol for :class:`GraphTrainableFeaturesPlot`-compatible plot functions.

    The callback extracts the pieces the plot function actually needs from the
    training :class:`~pytorch_lightning.LightningModule` — so plug-in
    functions receive **already-resolved** graph artifacts and never touch
    the raw model. Implementations yield ``(figure, tag)`` pairs; one call
    may emit multiple figures under distinct logging tags (e.g. node
    features + edge features). Any additional keyword arguments bound via
    the ``plot_fn:`` YAML block (``_partial_: true``) are forwarded through
    ``**kwargs``.

    Parameters passed by the callback
    ---------------------------------
    dataset_name
        Name of the dataset currently being plotted (used to look up per-
        dataset node/edge features and to build the artifact tag).
    node_attributes
        :class:`~anemoi.models.layers.graph.NamedNodesAttributes` for
        coordinate/mesh lookups.
    node_trainable_tensors
        Mapping from node-set name to its trainable parameter tensor. Empty
        when the model has no trainable node attributes.
    edge_trainable_modules
        Mapping from ``(src_nodes, dst_nodes)`` to graph mapper module for
        every edge-set that carries a trainable parameter. Empty when the
        model has no trainable edge attributes (e.g. hierarchical models).
    """

    def __call__(
        self,
        dataset_name: str,
        *,
        node_attributes: "NamedNodesAttributes",
        node_trainable_tensors: dict[str, Tensor],
        edge_trainable_modules: dict[tuple[str, str], Any],
        q_extreme_limit: float = 0.05,
        settings: Any = None,
        **kwargs: Any,
    ) -> Iterable[tuple[Figure, str]]: ...



def get_node_trainable_tensors(node_attributes: NamedNodesAttributes) -> dict[str, Tensor]:
    """Extract trainable node tensors from node attributes.

    Parameters
    ----------
    node_attributes : NamedNodesAttributes
        Node attributes object.

    Returns
    -------
    dict[str, Tensor]
        Mapping from node-set name to its trainable parameter tensor.
    """
    return {name: tt.trainable for name, tt in node_attributes.trainable_tensors.items() if tt.trainable is not None}


def _resolve_edge_provider(provider: Any, dataset_name: str) -> Any:
    """Resolve an edge provider to the specific dataset variant if needed."""
    if provider is None:
        return None
    if isinstance(provider, (dict,)):
        if dataset_name in provider:
            return provider[dataset_name]
        return None
    # Also handle torch.nn.ModuleDict without importing torch.nn at module level
    if type(provider).__name__ == "ModuleDict":
        if dataset_name in provider:
            return provider[dataset_name]
        return None
    return provider


def _has_trainable_edge_params(provider: Any) -> bool:
    """Return True if *provider* holds a trainable parameter."""
    if provider is None:
        return False
    trainable_module = getattr(provider, "trainable", None)
    if trainable_module is None:
        return False
    # Graph providers have TrainableTensor -> .trainable;
    # parameter is nested as .trainable.trainable.
    trainable_parameter = getattr(trainable_module, "trainable", None)
    return trainable_parameter is not None


def get_edge_trainable_modules(model: Any, dataset_name: str) -> dict[tuple[str, str], Any]:
    """Extract trainable edge modules for a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model carrying encoder/processor/decoder graph providers.
    dataset_name : str
        Dataset name used to select per-dataset providers.

    Returns
    -------
    dict[tuple[str, str], Any]
        Mapping from ``(src_nodes, dst_nodes)`` to graph mapper module,
        for each edge-set that has a trainable parameter.
    """
    trainable_modules = {}
    provider_specs = (
        ("encoder_graph_provider", (dataset_name, model._graph_name_hidden)),
        ("decoder_graph_provider", (model._graph_name_hidden, dataset_name)),
        ("processor_graph_provider", (model._graph_name_hidden, model._graph_name_hidden)),
    )
    for provider_name, edge_key in provider_specs:
        provider = _resolve_edge_provider(getattr(model, provider_name, None), dataset_name)
        if _has_trainable_edge_params(provider):
            trainable_modules[edge_key] = provider
    return trainable_modules


def edge_plot(
    fig: Figure,
    ax: plt.Axes,
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    data: np.ndarray,
    cmap: str = "coolwarm",
    title: str | None = None,
) -> None:
    """Lat-lon line plot.

    Parameters
    ----------
    fig : Figure
        Figure object handle
    ax : matplotlib.axes
        Axis object handle
    src_coords : np.ndarray of shape (num_edges, 2)
        Source latitudes and longitudes.
    dst_coords : np.ndarray of shape (num_edges, 2)
        Destination latitudes and longitudes.
    data : np.ndarray of shape (num_edges, 1)
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "coolwarm".
    title : str, optional
        Title for plot, by default None
    """
    edge_lines = np.stack([src_coords, dst_coords], axis=1)
    lc = LineCollection(edge_lines, cmap=cmap, linewidths=1)
    lc.set_array(data)

    psc = ax.add_collection(lc)

    xmin, xmax = edge_lines[:, 0, 0].min(), edge_lines[:, 0, 0].max()
    ymin, ymax = edge_lines[:, 1, 1].min(), edge_lines[:, 1, 1].max()
    ax.set_xlim((xmin - 0.1, xmax + 0.1))
    ax.set_ylim((ymin - 0.1, ymax + 0.1))

    map_features.plot(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    fig.colorbar(psc, ax=ax)


def plot_graph_node_features(
    node_attributes: NamedNodesAttributes,
    trainable_tensors: dict[str, Tensor],
    datashader: bool = False,
) -> Figure:
    """Plot trainable graph node features.

    Parameters
    ----------
    node_attributes: NamedNodesAttributes
        Node attributes object
    trainable_tensors: dict[str, torch.Tensor]
        Node trainable tensors
    datashader: bool, optional
        Scatter plot, by default False

    Returns
    -------
    Figure
        Figure object handle
    """
    nrows = len(trainable_tensors)
    ncols = max(tt.shape[1] for tt in trainable_tensors.values())

    figsize = (ncols * 4, nrows * 3)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout="tight")

    for row, (mesh, trainable_tensor) in enumerate(trainable_tensors.items()):
        latlons = node_attributes.get_coordinates(mesh).cpu().numpy()
        node_features = trainable_tensor.cpu().detach().numpy()

        lat, lon = latlons[:, 0], latlons[:, 1]

        for i in range(ncols):
            ax_ = ax[row, i] if ncols > 1 else ax[row]
            single_plot(
                fig,
                ax_,
                lon=lon,
                lat=lat,
                data=node_features[..., i],
                title=f"{mesh} trainable feature #{i + 1}",
                datashader=datashader,
                transform=None,
            )

    return fig


def plot_graph_edge_features(
    node_attributes: NamedNodesAttributes,
    trainable_modules: dict[tuple[str, str], Tensor],
    q_extreme_limit: float = 0.05,
) -> Figure:
    """Plot trainable graph edge features.

    Parameters
    ----------
    node_attributes: NamedNodesAttributes
        Node attributes object
    trainable_modules: dict[tuple[str, str], torch.Tensor]
        Edge trainable tensors.
    q_extreme_limit : float, optional
        Plot top & bottom quantile of edges trainable values, by default 0.05 (5%).

    Returns
    -------
    Figure
        Figure object handle
    """
    nrows = len(trainable_modules)
    ncols = max(tt.trainable.trainable.shape[1] for tt in trainable_modules.values())
    figsize = (ncols * 4, nrows * 3)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout="tight")

    for row, ((src, dst), graph_mapper) in enumerate(trainable_modules.items()):
        src_coords = node_attributes.get_coordinates(src).cpu().numpy()
        dst_coords = node_attributes.get_coordinates(dst).cpu().numpy()
        edge_index = graph_mapper.edge_index_base.cpu().numpy()
        edge_features = graph_mapper.trainable.trainable.cpu().detach().numpy()

        for i in range(ncols):
            ax_ = ax[row, i] if ncols > 1 else ax[row]
            feature = edge_features[..., i]

            top_perc = np.quantile(feature, 1 - q_extreme_limit)
            bottom_perc = np.quantile(feature, q_extreme_limit)
            mask = (feature >= top_perc) | (feature <= bottom_perc)

            edge_plot(
                fig,
                ax_,
                src_coords[edge_index[0, mask]][:, ::-1],
                dst_coords[edge_index[1, mask]][:, ::-1],
                feature[mask],
                title=f"{src} -> {dst} trainable feature #{i + 1}",
            )

    return fig


def graph_plot_fn(
    dataset_name: str,
    *,
    node_attributes: NamedNodesAttributes,
    node_trainable_tensors: dict[str, Tensor],
    edge_trainable_modules: dict[tuple[str, str], Any],
    q_extreme_limit: float = 0.05,
    settings=None,
    **_kwargs,
):
    """Default plug-in function for :class:`GraphTrainableFeaturesPlot`.

    Yields ``(figure, tag)`` pairs. Receives already-resolved graph
    artifacts from the callback (see :class:`GraphPlotFn` for the full
    contract). Emits warnings when the corresponding artifacts are empty.
    """
    datashader = getattr(settings, "datashader", True)

    if node_trainable_tensors.get(dataset_name) is not None:
        fig = plot_graph_node_features(
            node_attributes,
            node_trainable_tensors,
            datashader=datashader,
        )
        yield fig, f"node_trainable_params_{dataset_name}"
    else:
        LOGGER.warning("There are no trainable node attributes to plot.")

    if edge_trainable_modules:
        fig = plot_graph_edge_features(
            node_attributes,
            edge_trainable_modules,
            q_extreme_limit=q_extreme_limit,
        )
        yield fig, f"edge_trainable_params_{dataset_name}"
    else:
        LOGGER.warning("There are no trainable edge attributes to plot.")

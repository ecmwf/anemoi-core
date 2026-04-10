# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from torch import Tensor

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.training.diagnostics.evaluation.geospatial.maps import map_features
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import _hide_axes_ticks


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

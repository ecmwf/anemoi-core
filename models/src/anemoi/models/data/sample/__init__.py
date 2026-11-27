# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The contract between a data reader and :meth:`Batch.collate`.

A reader returns one :class:`BaseSample` per dataset and sample. The concrete class
says what kind of data it is, and knows how to collate a list of its own kind into
the matching :class:`~anemoi.models.data.sources.Source`.

Each kind is registered in :data:`sample_registry` under its name (``"gridded"``,
``"tabular"``). Both the data readers (training) and
:meth:`AnemoiModelInterface.predict_step` (inference) build their samples with
:func:`create_batched_struct`, which validates the inputs and converts them to the
canonical form.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import BaseTemplate
from anemoi.utils.registry import Registry

sample_registry = Registry(__name__)

# Imported after the registry is defined: each kind registers itself on import.
from anemoi.models.data.sample.base import BaseSample  # noqa: E402
from anemoi.models.data.sample.base import host_float32  # noqa: E402
from anemoi.models.data.sample.gridded import GriddedSample  # noqa: E402
from anemoi.models.data.sample.tabular import TabularSample  # noqa: E402

__all__ = [
    "GriddedSample",
    "BaseSample",
    "TabularSample",
    "create_batched_struct",
    "sample_registry",
]


def _layout_with_axes(
    layout: TensorLayout | Sequence[str],
    sample_cls: type[BaseSample],
    data_type: str,
) -> TensorLayout:
    """Return ``layout`` as a :class:`TensorLayout`, checking that it names the kind's required axes.

    The required axes are checked against the names given: :meth:`TensorLayout.from_tuple`
    places ``grid`` and ``variables`` even when they are not named.
    """
    names = layout.axis_names if isinstance(layout, TensorLayout) else tuple(layout)
    missing = [name for name in sample_cls.REQUIRED_AXES if name not in names]
    if missing:
        msg = (
            f"A {data_type!r} layout must name the axes {list(sample_cls.REQUIRED_AXES)}; {names} lacks {missing}. "
            f"It may leave out {list(sample_cls.OPTIONAL_AXES)}, which then have size 1."
        )
        raise ValueError(msg)
    return layout if isinstance(layout, TensorLayout) else TensorLayout.from_tuple(*names)


def create_batched_struct(
    *,
    data_type: str,
    variables: Sequence[str],
    layout: TensorLayout | Sequence[str],
    latitudes: Any,
    longitudes: Any,
    values: torch.Tensor | None = None,
    statistics: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    **kwargs: Any,
) -> BaseSample | BaseTemplate:
    """Validate the inputs and build the sample, or template, of the kind registered as ``data_type``.

    This is the single entry point used by the data readers and by
    :meth:`~anemoi.models.interface.AnemoiModelInterface.predict_step`. The fields
    common to every kind are checked and converted here; the rest is handed to the
    kind's :meth:`BaseSample.from_validated` or, without ``values``, to its
    :meth:`BaseSample.template_from_validated`.

    Parameters
    ----------
    data_type : str
        Name of the kind in :data:`sample_registry`, e.g. ``"gridded"`` or ``"tabular"``.
    variables : Sequence[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout or Sequence[str]
        The per-sample layout (no batch axis), or its axis names in order. It names
        exactly the axes of ``values``. It must name the kind's
        :attr:`~BaseSample.REQUIRED_AXES` and may leave out its
        :attr:`~BaseSample.OPTIONAL_AXES` (e.g. ``ensemble``), which are then inserted
        with size 1. Any axis order is accepted: ``values`` is put in the canonical order
        of :attr:`TensorLayout.AXES`, and the sample or template carries the canonical
        layout, optional axes included.
    latitudes, longitudes : array-like
        ``(N,)`` coordinates in **degrees**; converted to radians here.
    values : torch.Tensor, optional
        The sample's values. ``None`` to build a template that only describes the
        geometry, e.g. an inference target.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis.
    device : torch.device, optional
        Device of a template's geometry; a sample's follows its ``values``.
    **kwargs
        Arguments specific to the kind: ``grid_size`` (samples only) and ``shard_sizes``
        for gridded data; ``timedeltas``, ``boundaries`` (slices or ``(start, stop)``
        pairs) and ``shard_sizes`` for tabular data. Every kind also takes the optional
        ``time_offsets`` (one ``datetime.timedelta`` or ``numpy.timedelta64`` per time step,
        from the reference time) and ``reference_date`` (the absolute reference time).
        A template also needs the dataset ``name`` and the ``batch_size``,
        ``ensemble_size`` and ``time_size`` of the batch to decode. A tabular template
        repeats the geometry for each of the ``batch_size`` samples, and takes its time
        windows from ``boundaries`` rather than ``time_size``.

    Returns
    -------
    BaseSample or BaseTemplate
        The sample of the registered kind or, without ``values``, the kind's
        :attr:`BaseSample.template_type`, built from the same validated geometry.
    """
    sample_cls = sample_registry.lookup(data_type)

    device = values.device if values is not None else device

    # Degrees to radians in float64, rounded once to float32, as the graph builders do.
    latitudes = host_float32(latitudes, np.deg2rad)
    longitudes = host_float32(longitudes, np.deg2rad)
    if latitudes.shape != longitudes.shape:
        msg = (
            f"latitudes {tuple(latitudes.shape)} and longitudes {tuple(longitudes.shape)} "
            "must describe the same points."
        )
        raise ValueError(msg)
    coordinates = torch.from_numpy(np.stack((latitudes, longitudes), axis=-1)).to(device)
    num_points = coordinates.shape[0]

    layout = _layout_with_axes(layout, sample_cls, data_type)
    variables = list(variables)

    # Callers may order the axes freely and leave out the optional ones; everything downstream
    # sees the canonical order with every optional axis present.
    internal = TensorLayout.from_tuple(
        *(name for name in TensorLayout.AXES if layout.has_axis(name) or name in sample_cls.OPTIONAL_AXES),
    )
    if values is not None:
        conformed = layout.conform_to(values, internal)
        # Only a reordered tensor is copied; a reader's tensor already in this layout is kept as is.
        values = values if conformed is values else conformed.contiguous()
    layout = internal

    if values is None:
        return sample_cls.template_from_validated(
            n_points=num_points,
            device=device,
            variables=variables,
            layout=layout,
            coordinates=coordinates,
            statistics=statistics,
            **kwargs,
        )

    n_vars = values.shape[layout.axis("variables", ndim=values.ndim)]
    if n_vars != len(variables):
        msg = f"values carries {n_vars} variables but {len(variables)} names were given."
        raise ValueError(msg)
    n_grid = values.shape[layout.axis("grid", ndim=values.ndim)]
    if n_grid != num_points:
        msg = f"values carries {n_grid} points but {num_points} coordinates were given."
        raise ValueError(msg)

    return sample_cls.from_validated(
        n_points=num_points,
        device=device,
        values=values,
        variables=variables,
        layout=layout,
        coordinates=coordinates,
        statistics={} if statistics is None else statistics,
        **kwargs,
    )

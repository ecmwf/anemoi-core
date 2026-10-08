# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The contract between a data reader and :meth:`Batch.collate`.

A reader returns one :class:`SourceSample` per dataset and sample. The concrete class
says what kind of data it is, and knows how to collate a list of its own kind into
the matching :class:`~anemoi.models.data.sources.Source`.

Each kind is registered in :data:`sample_registry` under its name (``"gridded"``,
``"tabular"``). Both the data readers (training) and
:meth:`AnemoiModelInterface.predict_step` (inference) build their samples with
:func:`create_source_sample`, which validates the inputs and converts them to the
canonical form.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
from anemoi.utils.registry import Registry

from anemoi.models.data.layout import TensorLayout

sample_registry: Registry = Registry(__name__)

# Imported after the registry is defined: each kind registers itself on import.
from anemoi.models.data.sample.base import SourceSample  # noqa: E402
from anemoi.models.data.sample.gridded import GriddedSourceSample  # noqa: E402
from anemoi.models.data.sample.tabular import TabularSourceSample  # noqa: E402

__all__ = [
    "GriddedSourceSample",
    "SourceSample",
    "TabularSourceSample",
    "create_source_sample",
    "sample_registry",
]


def create_source_sample(
    *,
    data_type: str,
    variables: Sequence[str],
    layout: TensorLayout | Sequence[str],
    latitudes: Any,
    longitudes: Any,
    data: torch.Tensor | None = None,
    statistics: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SourceSample:
    """Validate the inputs and build the :class:`SourceSample` registered as ``data_type``.

    This is the single entry point used by the data readers and by
    :meth:`~anemoi.models.interface.AnemoiModelInterface.predict_step`. The fields
    common to every kind are checked and converted here; the rest is handed to the
    kind's :meth:`SourceSample.from_validated`.

    Parameters
    ----------
    data_type : str
        Name of the kind in :data:`sample_registry`, e.g. ``"gridded"`` or ``"tabular"``.
    variables : Sequence[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout or Sequence[str]
        The per-sample layout (no batch axis), or its axis names in order.
    latitudes, longitudes : array-like
        ``(N,)`` coordinates in **degrees**; converted to radians here.
    data : torch.Tensor, optional
        The sample's data. ``None`` for a sample that only describes its geometry.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis.
    **kwargs
        Arguments specific to the kind: ``grid_size`` and ``shard_sizes`` for gridded
        samples; ``timedeltas``, ``boundaries`` (slices or ``(start, stop)`` pairs)
        and ``shard_sizes`` for tabular samples.
    """
    sample_type = sample_registry.lookup(data_type)
    device = data.device if data is not None else None

    latitudes = _as_radians(latitudes, device)
    longitudes = _as_radians(longitudes, device)
    if latitudes.shape != longitudes.shape:
        msg = (
            f"latitudes {tuple(latitudes.shape)} and longitudes {tuple(longitudes.shape)} "
            "must describe the same points."
        )
        raise ValueError(msg)
    n_points = latitudes.shape[0]

    if not isinstance(layout, TensorLayout):
        layout = TensorLayout.from_tuple(*layout)
    variables = list(variables)

    if data is not None:
        if data.ndim != layout.ndim:
            msg = f"data of shape {tuple(data.shape)} does not match the layout {layout!r}."
            raise ValueError(msg)
        n_vars = data.shape[layout.axis("variables", ndim=data.ndim)]
        if n_vars != len(variables):
            msg = f"data carries {n_vars} variables but {len(variables)} names were given."
            raise ValueError(msg)
        n_grid = data.shape[layout.axis("grid", ndim=data.ndim)]
        if n_grid != n_points:
            msg = f"data carries {n_grid} points but {n_points} coordinates were given."
            raise ValueError(msg)

    return sample_type.from_validated(
        n_points=n_points,
        device=device,
        data=data,
        variables=variables,
        layout=layout,
        latitudes=latitudes,
        longitudes=longitudes,
        statistics={} if statistics is None else statistics,
        **kwargs,
    )


def _as_radians(degrees: Any, device: torch.device | None) -> torch.Tensor:
    """Return ``degrees`` as a flat float32 tensor in radians."""
    return torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32, device=device).reshape(-1))

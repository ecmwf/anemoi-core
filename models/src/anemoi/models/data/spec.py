# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-source metadata that does not change from one batch to the next."""

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import torch

from anemoi.models.data.layout import TensorLayout

if TYPE_CHECKING:
    from anemoi.models.data.sources.base import Source

LOGGER = logging.getLogger(__name__)


def fancy_variable_index(
    indices: slice | Sequence[int] | torch.Tensor,
) -> slice | list[int] | Sequence[int] | torch.Tensor:
    """Return a variable index safe for dimension-preserving indexing.

    Statistics are stored as numpy arrays, so a tensor index is converted
    to a plain list of ints (preserving the variable axis).

    Slices and other sequence indices are returned unchanged.
    """
    if isinstance(indices, torch.Tensor):
        return indices.tolist()
    return indices


@dataclass(frozen=True)
class SourceSpec(ABC):
    """Metadata describing one source dataset, independent of any single batch.

    A spec is built once per dataset per run and shared by every
    :class:`~anemoi.models.data.source.Source` over that dataset. Splitting it
    out of the view means the derived lookups it owns (:attr:`name_to_index`) are
    computed once rather than on every ``batch[name]`` access, and that a caller
    can describe a source to ``model.forward()`` without carrying its data - see
    :meth:`empty`.

    Parameters
    ----------
    name : str
        Dataset name, as keyed in :class:`~anemoi.models.data.batch.Batch`.
    variables : list[str]
        Variable names along the layout's ``variables`` axis, in order. Must be
        unique.
    layout : TensorLayout
        Mapping from logical axes to physical dimension positions.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis (``mean``, ``stdev``, ...), as
        produced by ``anemoi-datasets``. Values are normally :class:`numpy.ndarray`
        but torch tensors are accepted.
    grid_size : int or None, optional
        Full grid size before any distributed sharding. ``None`` for observation
        datasets, which have no static grid.
    coordinates_are_static : bool, optional
        Whether this dataset's coordinate tensor is fixed for the whole run and so
        may be shared by reference rather than transferred per batch.
    metadata : Mapping[str, Any], optional
        Free-form per-source metadata (e.g. ``dataset.metadata``, per-variable
        metadata). Not interpreted here.

    Notes
    -----
    This dataclass is deliberately **not** ``slots=True``: :func:`cached_property`
    needs an instance ``__dict__`` to memoise into.
    """

    name: str
    variables: list[str]
    layout: TensorLayout
    statistics: Mapping[str, Any] = field(default_factory=dict)
    grid_size: int | None = None
    coordinates_are_static: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the metadata that does not depend on a materialized tensor."""
        if self.variables is None or len(set(self.variables)) != len(self.variables):
            raise ValueError(f"Source {self.name!r} requires unique variable names.")

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """Mapping from variable name to index along the variables axis."""
        return {name: idx for idx, name in enumerate(self.variables)}

    @property
    def n_variables(self) -> int:
        """Number of variables along the variables axis."""
        return len(self.variables)

    @abstractmethod
    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "SourceSpec":
        """Return a new spec restricted to the given variable indices.

        Both :attr:`variables` and :attr:`statistics` are indexed, so they stay
        consistent with the data tensor the caller indexes alongside.
        """
        pass

    @property
    @abstractmethod
    def is_tabular(self) -> bool:
        """Whether time is folded into the grid axis (sparse observation sources)."""
        pass

    def clone(self, **kwargs) -> "SourceSpec":
        """Return a new spec with replacements, sharing fields that are not replaced."""
        return replace(self, **kwargs)

    def empty(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Source":
        """Return a source carrying this spec and no data.

        The payload has the full variable axis but a zero-length grid axis, and a
        size of one along ``time`` and ``ensemble``. This is the "batch without the
        data" form: enough for a consumer to read variables, layout and statistics
        off a source it is being asked to produce, without materializing it.

        Parameters
        ----------
        batch_size : int, optional
            Size of the batch axis (for tabular sources, the number of per-sample
            tensors), by default 1.
        device : torch.device or str, optional
            Device for the empty tensors, by default the current default device.
        dtype : torch.dtype, optional
            Dtype for the empty tensors, by default ``torch.float32``.

        Returns
        -------
        Source
            A :class:`~anemoi.models.data.source.GriddedSource` or
            :class:`~anemoi.models.data.source.TabularSource` matching this
            spec's layout.
        """
        # Local import: views.py imports this module, so importing it at module
        # scope would be circular.
        from anemoi.models.data.sources import make_source

        layout = self.layout.normalized(self.layout.ndim)
        sizes = {"batch": batch_size, "time": 1, "ensemble": 1, "grid": 0, "variables": self.n_variables}
        shape = [0] * layout.ndim
        for axis_name in layout.dims:
            shape[getattr(layout, axis_name)] = sizes[axis_name]

        if self.is_tabular:
            data = [torch.empty(shape, dtype=dtype, device=device) for _ in range(batch_size)]
            coordinates = [torch.empty((0, 2), dtype=dtype, device=device) for _ in range(batch_size)]
            boundaries = [() for _ in range(batch_size)]
        else:
            data = torch.empty(shape, dtype=dtype, device=device)
            coordinates = torch.empty((0, 2), dtype=dtype, device=device)
            boundaries = None

        return make_source(spec=self, data=data, coordinates=coordinates, boundaries=boundaries)


class GriddedSpec(SourceSpec):

    @property
    def is_tabular(self) -> bool:
        return False

    def select_variables(self, indices: list[int]) -> "SourceSpec":
        new_variables = [self.variables[i] for i in indices]
        new_statistics = {key: value[fancy_variable_index(indices)] for key, value in self.statistics.items()}
        return self.clone(variables=new_variables, statistics=new_statistics)


class TabularSpec(SourceSpec):

    @property
    def is_tabular(self) -> bool:
        return True

    def select_variables(self, indices: list[int]) -> "SourceSpec":
        new_variables = [self.variables[i] for i in indices]
        new_statistics = {key: value[fancy_variable_index(indices)] for key, value in self.statistics.items()}
        return self.clone(variables=new_variables, statistics=new_statistics)


def make_spec(layout: TensorLayout, **kwargs) -> "SourceSpec":
    if layout.time_in_grid:
        return TabularSpec(layout=layout, **kwargs)

    return GriddedSpec(layout=layout, **kwargs)

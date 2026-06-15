# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from functools import cached_property
import einops
import numpy as np
import torch

from anemoi.models.data.tensor_layout import TensorLayout

import logging


LOGGER = logging.getLogger(__name__)


def create_source_view(**kwargs) -> "SourceView":
    """Factory function to create a SourceView for a source dataset."""
    if kwargs.pop("is_static"):
        return GriddedSourceView(**kwargs)

    return TabularSourceView(**kwargs)


@dataclass(frozen=True, slots=True)
class FlattenView:
    """A flattened view of the data, coordinates and timedeltas for a single sample.

    This is used as an intermediate representation when applying functions or
    losses to the data, before unflattening back to a SourceView.
    """

    data: torch.Tensor
    coordinates: torch.Tensor | None
    device: torch.device | None


@dataclass(frozen=True, slots=True)
class SourceView(ABC):
    """Per-dataset view returned by :meth:`Batch.view`.

    Bundles the per-dataset payload (data, coordinates, timedeltas) with
    its :class:`TensorLayout` so callers can index logical axes (``time``,
    ``variables``) without hard-coded dimension positions. The same API
    works for gridded and sparse observation datasets thanks to the
    ``layout.time_in_grid`` dispatch.
    """

    name: str
    data: torch.Tensor | list[torch.Tensor]
    variables: list[str]
    statistics: dict[str, torch.Tensor]
    coordinates: torch.Tensor | list[torch.Tensor] | None
    layout: TensorLayout
    timedeltas: torch.Tensor | list[torch.Tensor] | None = None
    grid_shard_indices: Any = None
    boundaries: list[tuple[slice, ...]] | None = None

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """Mapping from variable name to index along the variables axis."""
        return {name: idx for idx, name in enumerate(self.variables)}

    def clone(self, **kwargs) -> "SourceView":
        """Return a deep copy of this view (clones the data tensor)."""
        return replace(self, **kwargs)

    def select(self, **kwargs) -> "SourceView":
        """Return a new view restricted to the given indices along logical dimensions.
        
        Example
        -------
        >>> view.select(time=slice(0, 10), variables=[0, 2])
        """
        source = self
        for dim, indices in kwargs.items():
            if dim == "time":
                source = source.select_time(indices)
            elif dim == "variables":
                source = source.select_variables(indices)
            else:
                raise ValueError(f"Unsupported dimension for selection: {dim!r}. Supported dimensions are 'time' and 'variables'.")
        return source

    @abstractmethod
    def flatten(self) -> FlattenView:
        """Return a flattened view of the data, coordinates and timedeltas for a single sample."""
        pass 

    @abstractmethod
    def select_time(self, indices: slice | Sequence[int] | int) -> "SourceView":
        """Return a new view restricted to the given time indices."""
        pass

    @abstractmethod
    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "SourceView":
        """Return a new view restricted to the given variable indices."""
        pass

    @abstractmethod
    def unflatten_data_2d(self, data_2d: torch.Tensor) -> "SourceView":
        """Unflatten a 2D data tensor back to the original grid shape."""
        pass

    @abstractmethod
    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "SourceView":
        """Apply a function to this view, returning a new view with the same metadata."""
        pass

    @abstractmethod
    def apply_loss(self, other: "SourceView", loss_func: Callable, **kwargs) -> "SourceView":
        """Apply a loss function to this view and another view, returning the result."""
        pass

    def _time_axis_size(self) -> int:
        """Return the logical number of time steps in this view."""
        if self.layout.time_in_grid:
            if self.boundaries is None:
                msg = "Sparse view has no 'boundaries' metadata; cannot determine time size."
                raise ValueError(msg)
            return len(self.boundaries[0]) if self.boundaries else 0
        if self.layout.time is None:
            msg = f"Layout {self.layout!r} has no time axis."
            raise ValueError(msg)
        assert isinstance(self.data, torch.Tensor)
        return self.data.shape[self.layout.time]

    def _index_vars(self, tensor: torch.Tensor, indices: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        var_dim = self.layout.axis("variables", ndim=tensor.ndim)
        if isinstance(indices, slice):
            slicer: list[Any] = [slice(None)] * tensor.ndim
            slicer[var_dim] = indices
            return tensor[tuple(slicer)]
        idx = torch.as_tensor(list(indices) if not isinstance(indices, torch.Tensor) else indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(var_dim, idx)


class GriddedSourceView(SourceView):
    """SourceView for gridded datasets, where time is an explicit dimension."""

    is_static: bool = True
    pattern_for_2d: str = "(batch ensemble grid) (time variables)"

    def __post_init__(self):
        if self.layout.time is None:
            msg = f"{self.__class__.__name__} requires a layout with a time axis; got {self.layout!r}."
            raise ValueError(msg)
        if isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a single tensor, not a list."
            raise TypeError(msg)

    @property
    def device(self) -> torch.device:
        assert isinstance(self.data, torch.Tensor), f"{self.__class__.__name__} data must be a single tensor."
        return self.data.device

    def flatten(self) -> FlattenView:
        current_pattern = self.layout.pattern
        flattened_data = einops.rearrange(self.data, f"{current_pattern} -> {self.pattern_for_2d}")
        device = self.data.device

        batch_size = self.data.shape[self.layout.batch]
        coordinates = einops.repeat(self.coordinates, "grid latlon -> (batch grid) latlon", batch=batch_size)

        return FlattenView(
            data=flattened_data,
            coordinates=coordinates.to(device),
            device=device,
        )

    @property
    def ndim(self) -> int:
        assert isinstance(self.data, torch.Tensor), f"{self.__class__.__name__} data must be a single tensor."
        return self.data.ndim

    def unflatten_data_2d(self, data_2d: torch.Tensor) -> "GriddedSourceView":
        target_layout = self.layout.pattern
        batch_size = self.data.shape[self.layout.batch]
        ensemble_size = self.data.shape[self.layout.ensemble]
        num_out_times = self.data.shape[self.layout.time]
        new_data = einops.rearrange(
            data_2d,
            f"{self.pattern_for_2d} -> {target_layout}",
            batch=batch_size,
            ensemble=ensemble_size,
            time=num_out_times
        )
        return self.clone(data=new_data)

    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "GriddedSourceView":
        """Apply a function to this view, returning a new view with the same metadata."""
        new_data = func(
            self.data if in_place else self.data.clone(),
            statistics=self.statistics,
            name_to_index=self.name_to_index,
            **kwargs,
        )
        return self.clone(data=new_data)

    def apply_loss(self, other: "GriddedSourceView", loss_func: Callable, **kwargs) -> torch.Tensor:
        """Apply a loss function to this view and another view, returning the result."""
        assert isinstance(other, GriddedSourceView), f"Other view must be a GriddedSourceView; got {type(other).__name__}."
        assert self.layout == other.layout, f"Both views must have the same layout; got {self.layout!r} and {other.layout!r}."
        #assert self.variables == other.variables, f"Both views must have the same variables; got {self.variables} and {other.variables}."
        assert torch.all(self.coordinates == other.coordinates), f"Both views must have the same coordinates; got {self.coordinates} and {other.coordinates}."
        return loss_func(
            self.data,
            other.data,
            layout=self.layout,
            statistics=self.statistics,
            name_to_index=self.name_to_index,
            **kwargs
        )

    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "GriddedSourceView":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        new_data = self._index_vars(self.data, indices)
        new_variables = self.variables[indices] if isinstance(indices, slice) else [self.variables[i] for i in indices]
        new_statistics = {k: v[indices] for k, v in self.statistics.items()}
        return self.clone(data=new_data, variables=new_variables, statistics=new_statistics)

    def index_select(self, dim: int, index: torch.Tensor) -> "GriddedSourceView":
        """Return a new view with the data tensor indexed along a given dimension."""
        assert isinstance(self.data, torch.Tensor), f"{self.__class__.__name__} data must be a single tensor."
        new_data = self.data.index_select(dim, index)
        return self.clone(data=new_data)

    def select_time(self, indices: "slice | Sequence[int] | int") -> "GriddedSourceView":
        """Return a new view restricted to the given time indices.

        Parameters
        ----------
        indices : int, slice, or sequence of int
            Positions along the logical *time* axis. For gridded datasets
            this indexes ``layout.time`` directly. For sparse observation
            datasets it picks the corresponding boundary slices from
            ``boundaries`` (per sample), updating data, coordinates,
            timedeltas and the boundary list consistently.

        Returns
        -------
        SourceView
            A new view with the same :class:`TensorLayout` but reduced
            time extent.
        """
        if isinstance(indices, slice):
            time_size = self._time_axis_size()
            idx_list = list(range(*indices.indices(time_size)))
        elif isinstance(indices, int):
            idx_list = [int(indices)]
        else:
            idx_list = [int(i) for i in indices]

        if self.layout.time is None:
            msg = f"Layout {self.layout!r} has no time axis; cannot select_time on a gridded view."
            raise ValueError(msg)

        assert isinstance(self.data, torch.Tensor), "Gridded view must wrap a single tensor."
        idx = torch.as_tensor(idx_list, dtype=torch.long, device=self.data.device)
        new_data = self.data.index_select(self.layout.time, idx)
        return self.clone(data=new_data)


class TabularSourceView(SourceView):
    """SourceView for tabular datasets, where time is represented by boundaries."""

    is_static: bool = False

    def __post_init__(self):
        if not self.layout.time_in_grid:
            msg = f"TabularSourceView requires a layout with time_in_grid=True; got {self.layout!r}."
            raise ValueError(msg)

        if not isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a list of tensors, not a single tensor."
            raise TypeError(msg)

    @property
    def ndim(self) -> int:
        assert isinstance(self.data, list), f"{self.__class__.__name__} data must be a list of tensors."
        assert isinstance(self.data[0], torch.Tensor), f"{self.__class__.__name__} data must be a list of tensors."
        return self.data[0].ndim

    @property
    def device(self) -> torch.device:
        assert isinstance(self.data, list) and len(self.data) > 0, f"{self.__class__.__name__} data must be a non-empty list of tensors."
        return self.data[0].device

    def flatten(self) -> FlattenView:
        if len(self.data) > 1:
            data = torch.cat(self.data, dim=0)
            coordinates = torch.cat(self.coordinates, dim=0)
        elif len(self.data) == 1:
            data = self.data[0]
            coordinates = self.coordinates[0]

        device = data.device
        coordinates = coordinates.to(device)

        return FlattenView(
            data=data,
            coordinates=coordinates.to(device),
            device=device,
        )

    def unflatten_data_2d(self, data_2d: torch.Tensor) -> "TabularSourceView":
        assert isinstance(self.data, list), f"{self.__class__.__name__} data must be a list of tensors."
        batch_sizes = [data.shape[self.layout.grid] for data in self.data]
        batch_starts = np.cumsum([0] + batch_sizes[:-1])
        new_data = [data_2d.narrow(self.layout.grid, int(batch_starts[i]), length) for i, length in enumerate(batch_sizes)]
        return self.clone(data=new_data)

    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "TabularSourceView":
        """Apply a function to this view, returning a new view with the same metadata."""
        new_data = [
            func(
                data if in_place else data.clone(),
                statistics=self.statistics,
                name_to_index=self.name_to_index,
                **kwargs,
            ) for data in self.data
        ]
        return self.clone(data=new_data)

    def apply_loss(self, other: "TabularSourceView", loss_func: Callable, **kwargs) -> torch.Tensor:
        """Apply a loss function to this view and another view, returning the result."""
        assert isinstance(other, TabularSourceView), f"Other view must be a TabularSourceView; got {type(other).__name__}."
        assert self.layout == other.layout, f"Both views must have the same layout; got {self.layout!r} and {other.layout!r}."
        assert len(self.data) == len(other.data), f"Both views must have the same number of samples; got {len(self.data)} and {len(other.data)}."
        #assert self.variables == other.variables, f"Both views must have the same variables; got {self.variables} and {other.variables}."

        losses = []
        for i, (pred, target) in enumerate(zip(self.data, other.data)):
            assert pred.shape == target.shape, f"Sample {i} of both views must have the same shape; got {pred.shape} and {target.shape}."
            assert torch.all(self.coordinates[i] == other.coordinates[i]), f"Sample {i} of both views must have the same coordinates; got {self.coordinates[i]} and {other.coordinates[i]}."

            losses.append(
                loss_func(
                    pred,
                    target,
                    layout=self.layout,
                    statistics=self.statistics,
                    name_to_index=self.name_to_index,
                    **kwargs
                )
            )
    
        return torch.mean(*tuple(losses))

    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "TabularSourceView":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        new_data = [self._index_vars(t, indices) for t in self.data]
        new_variables = self.variables[indices] if isinstance(indices, slice) else [self.variables[i] for i in indices]
        new_statistics = {k: v[indices] for k, v in self.statistics.items()}
        return self.clone(data=new_data, variables=new_variables, statistics=new_statistics)

    def select_time(self, indices: "slice | Sequence[int] | int") -> "TabularSourceView":
        """Return a new view restricted to the given time indices.

        Parameters
        ----------
        indices : int, slice, or sequence of int
            Positions along the logical *time* axis. For gridded datasets
            this indexes ``layout.time`` directly. For sparse observation
            datasets it picks the corresponding boundary slices from
            ``boundaries`` (per sample), updating data, coordinates,
            timedeltas and the boundary list consistently.

        Returns
        -------
        SourceView
            A new view with the same :class:`TensorLayout` but reduced
            time extent.
        """
        if isinstance(indices, slice):
            time_size = self._time_axis_size()
            idx_list = list(range(*indices.indices(time_size)))
        elif isinstance(indices, int):
            idx_list = [int(indices)]
        else:
            idx_list = [int(i) for i in indices]

        if self.boundaries is None:
            msg = "Sparse view has no 'boundaries' metadata, cannot select_time."
            raise ValueError(msg)

        assert isinstance(self.data, list), f"{self.__class__.__name__} must wrap a list[Tensor]."

        new_data = []
        new_coords = []
        new_timedeltas = []
        new_boundaries = []

        for sample_idx, sample_bounds in enumerate(self.boundaries):
            selected_slices = [sample_bounds[t] for t in idx_list]
            sample_data = self.data[sample_idx]
            data_pieces = [sample_data.narrow(self.layout.grid, s.start, s.stop - s.start) for s in selected_slices]
            new_data.append(torch.cat(data_pieces, dim=self.layout.grid) if data_pieces else sample_data.narrow(self.layout.grid, 0, 0))

            if new_coords is not None and self.coordinates is not None:
                sample_coords = self.coordinates[sample_idx]
                coord_pieces = [sample_coords[s.start : s.stop] for s in selected_slices]
                new_coords.append(
                    torch.cat(coord_pieces, dim=0) if coord_pieces else sample_coords[:0],
                )

            if new_timedeltas is not None and self.timedeltas is not None:
                sample_td = self.timedeltas[sample_idx]
                td_pieces = [sample_td[s.start : s.stop] for s in selected_slices]
                new_timedeltas.append(
                    torch.cat(td_pieces, dim=0) if td_pieces else sample_td[:0],
                )

            offset = 0
            compact = []
            for s in selected_slices:
                length = s.stop - s.start
                compact.append(slice(offset, offset + length))
                offset += length
            new_boundaries.append(tuple(compact))

        return self.clone(
            data=new_data,
            coordinates=new_coords if new_coords else self.coordinates,
            timedeltas=new_timedeltas if new_timedeltas else self.timedeltas,
            boundaries=new_boundaries,
        )

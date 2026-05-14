# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from anemoi.models.data.tensor_layout import TensorLayout
import torch
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class DatasetView:
    """Per-dataset view returned by :meth:`Batch.view`.

    Bundles the per-dataset payload (data, coordinates, timedeltas) with
    its :class:`TensorLayout` so callers can index logical axes (``time``,
    ``variables``) without hard-coded dimension positions. The same API
    works for gridded and sparse observation datasets thanks to the
    ``layout.time_in_grid`` dispatch.
    """

    name: str
    data: torch.Tensor | list[torch.Tensor]
    coordinates: torch.Tensor | list[torch.Tensor] | None
    is_static: bool
    layout: TensorLayout
    timedeltas: torch.Tensor | list[torch.Tensor] | None = None
    grid_shard_indices: Any = None
    boundaries: list[tuple[slice, ...]] | None = None

    def time_slices(self) -> list[tuple[slice, ...]] | None:
        """Return per-sample boundary slices for sparse views; ``None`` otherwise.

        For sparse observation datasets (``layout.time_in_grid=True``) the
        ``boundaries`` describe which contiguous ranges of the grid axis
        belong to each time step; for gridded datasets time is an explicit
        dim so this returns ``None``.
        """
        return self.boundaries if self.layout.time_in_grid else None

    def select_time(self, indices: "slice | Sequence[int] | int") -> "DatasetView":
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
        DatasetView
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

        if not self.layout.time_in_grid:
            return self._select_time_gridded(idx_list)

        return self._select_time_sparse(idx_list)

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

    def _select_time_gridded(self, indices: list[int]) -> "DatasetView":
        if self.layout.time is None:
            msg = f"Layout {self.layout!r} has no time axis; cannot select_time on a gridded view."
            raise ValueError(msg)
        assert isinstance(self.data, torch.Tensor), "Gridded view must wrap a single tensor."
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.data.device)
        new_data = self.data.index_select(self.layout.time, idx)
        return dataclass_replace(self, data=new_data)

    def _select_time_sparse(self, indices: list[int]) -> "DatasetView":
        if self.boundaries is None:
            msg = "Sparse view has no 'boundaries' metadata; cannot select_time."
            raise ValueError(msg)
        assert isinstance(self.data, list), "Sparse view must wrap a list[Tensor]."

        coords_list = self.coordinates if isinstance(self.coordinates, list) else None
        if self.coordinates is not None and coords_list is None:
            msg = "Sparse view coordinates must be a list[Tensor]."
            raise TypeError(msg)
        td_list = self.timedeltas if isinstance(self.timedeltas, list) else None
        if self.timedeltas is not None and td_list is None:
            msg = "Sparse view timedeltas must be a list[Tensor]."
            raise TypeError(msg)

        grid_dim = self.layout.grid

        new_data: list[torch.Tensor] = []
        new_coords: list[torch.Tensor] | None = [] if coords_list is not None else None
        new_timedeltas: list[torch.Tensor] | None = [] if td_list is not None else None
        new_boundaries: list[tuple[slice, ...]] = []

        for sample_idx, sample_bounds in enumerate(self.boundaries):
            selected_slices = [sample_bounds[t] for t in indices]
            sample_data = self.data[sample_idx]
            data_pieces = [sample_data.narrow(grid_dim, s.start, s.stop - s.start) for s in selected_slices]
            new_data.append(torch.cat(data_pieces, dim=grid_dim) if data_pieces else sample_data.narrow(grid_dim, 0, 0))

            if new_coords is not None and coords_list is not None:
                sample_coords = coords_list[sample_idx]
                coord_pieces = [sample_coords[s.start : s.stop] for s in selected_slices]
                new_coords.append(
                    torch.cat(coord_pieces, dim=0) if coord_pieces else sample_coords[:0],
                )

            if new_timedeltas is not None and td_list is not None:
                sample_td = td_list[sample_idx]
                td_pieces = [sample_td[s.start : s.stop] for s in selected_slices]
                new_timedeltas.append(
                    torch.cat(td_pieces, dim=0) if td_pieces else sample_td[:0],
                )

            offset = 0
            compact: list[slice] = []
            for s in selected_slices:
                length = s.stop - s.start
                compact.append(slice(offset, offset + length))
                offset += length
            new_boundaries.append(tuple(compact))

        return dataclass_replace(
            self,
            data=new_data,
            coordinates=new_coords if new_coords is not None else self.coordinates,
            timedeltas=new_timedeltas if new_timedeltas is not None else self.timedeltas,
            boundaries=new_boundaries,
        )

    def select_vars(self, indices: Sequence[int] | torch.Tensor | slice) -> "DatasetView":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        if isinstance(self.data, list):
            new_data = [self._index_vars(t, indices) for t in self.data]
        else:
            new_data = self._index_vars(self.data, indices)
        return dataclass_replace(self, data=new_data)

    def _index_vars(self, tensor: torch.Tensor, indices: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        var_dim = self.layout.axis("variables", ndim=tensor.ndim)
        if isinstance(indices, slice):
            slicer: list[Any] = [slice(None)] * tensor.ndim
            slicer[var_dim] = indices
            return tensor[tuple(slicer)]
        idx = torch.as_tensor(list(indices) if not isinstance(indices, torch.Tensor) else indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(var_dim, idx)

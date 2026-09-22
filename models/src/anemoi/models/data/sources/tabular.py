# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.distributed import ProcessGroup

from anemoi.models.data.flat import FlatSource
from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources.base import _Source
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import check_shard_sizes_match_group
from anemoi.models.distributed.utils import model_is_distributed

LOGGER = logging.getLogger(__name__)


def _shape_without_ensemble_dim(tensor: torch.Tensor, layout: TensorLayout) -> tuple[int, ...]:
    """Tensor shape with the ensemble axis removed, needed for comparisons that ignore member count."""
    if layout.ensemble is None:
        return tuple(tensor.shape)
    axis = layout.axis("ensemble", ndim=tensor.ndim)
    return tuple(size for dim, size in enumerate(tensor.shape) if dim != axis)


def _fold_members(source: "TabularSource", sample: torch.Tensor) -> torch.Tensor:
    """Fold one sample's ensemble axis into its node axis, as the gridded path does.

    A gridded source flattens to ``(batch ensemble grid)``; doing the same here keeps
    the two kinds interchangeable downstream - the encoder and decoder graphs see one
    node set per ``(sample, member)`` either way.
    """
    layout = source.layout
    if layout.ensemble is None:
        return sample
    ensemble_axis = layout.axis("ensemble", ndim=sample.ndim)
    grid_axis = layout.axis("grid", ndim=sample.ndim)
    if ensemble_axis > grid_axis:
        msg = (
            f"Source {source.name!r} expects the ensemble axis before the grid axis so that "
            f"folding yields (ensemble grid) order; got {layout!r}."
        )
        raise ValueError(msg)
    return sample.flatten(ensemble_axis, grid_axis)


class TabularSource(_Source):
    """Tabular data source."""

    def __post_init__(self):
        super().__post_init__()
        if self.timedeltas is None:
            msg = f"{self.__class__.__name__} requires timedeltas to be provided; got None."
            raise ValueError(msg)

        ts = tuple(len(t) for t in self.timedeltas)
        cs = tuple(len(c) for c in self.coordinates)
        if ts != cs:
            msg = (
                f"{self.__class__.__name__} {self.name!r} timedeltas and coordinates must contain the same number of nodes, "
                f"got {sum(ts)} and {sum(cs)}."
            )
            raise ValueError(msg)

        if not self.layout.time_in_grid:
            msg = f"{self.__class__.__name__} requires a layout with time_in_grid=True; got {self.layout!r}."
            raise ValueError(msg)

        if not isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a list of tensors, not a single tensor."
            raise TypeError(msg)

        for sample, sample_coords in zip(self.data, self.coordinates, strict=True):
            if tuple(sample_coords.shape) != (sample.shape[self.layout.grid], 2):
                raise ValueError(f"Source {self.name!r} requires one latitude/longitude pair per node.")

    @property
    def ndim(self) -> int:
        return self.data[0].ndim

    @property
    def device(self) -> torch.device:
        return self.data[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.data[0].dtype

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members per sample; 1 when the layout has no ensemble axis."""
        if self.layout.ensemble is None:
            return 1
        return self.data[0].shape[self.layout.ensemble]

    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "TabularSource":
        """Apply a function to this view, returning a new view with the same metadata."""
        new_data = [
            func(
                data if in_place else data.clone(),
                statistics=self.statistics,
                name_to_index=self.name_to_index,
                **kwargs,
            )
            for data in self.data
        ]
        return self.clone(data=new_data)

    def flatten(self) -> FlatSource:
        folded = [_fold_members(self, sample) for sample in self.data]
        # coordinates and timedeltas are repeated per member to line up with the folded data
        repeated_coords = [c.repeat(self.ensemble_size, 1) for c in self.coordinates]
        repeated_timedeltas = (
            None if self.timedeltas is None else [td.repeat(self.ensemble_size) for td in self.timedeltas]
        )

        if len(folded) > 1:
            data = torch.cat(folded, dim=0)
            coordinates = torch.cat(repeated_coords, dim=0)
            timedeltas = None if repeated_timedeltas is None else torch.cat(repeated_timedeltas, dim=0)
        else:
            data = folded[0]
            coordinates = repeated_coords[0]
            timedeltas = None if repeated_timedeltas is None else repeated_timedeltas[0]

        # Flatten per-window shard sizes into one list for the concatenated data.
        # NOTE this changes the order of observations when gathering:
        #   GPU0  GPU1   GPU0  GPU1            GPU0        GPU1
        #   w1_0, w1_1 | w2_0, w2_1  becomes  w1_0, w2_0, w1_1, w2_1
        flat_shard_sizes = None
        if self.shard_sizes is not None:
            if len(self.shard_sizes) != 1:
                msg = (
                    f"Source {self.name!r}: a sharded tabular source is supported only at batch size 1, "
                    f"but this batch has {len(self.shard_sizes)} samples."
                )
                raise NotImplementedError(msg)
            window_shard_sizes = self.shard_sizes[0]
            # sum per-rank shard sizes across all windows to get totals for the concatenated data
            flat_shard_sizes = [
                sum(sizes[rank] for sizes in window_shard_sizes) for rank in range(len(window_shard_sizes[0]))
            ]

        batch_sizes = tuple(sample.shape[self.layout.grid] for sample in self.data for _ in range(self.ensemble_size))
        device = data.device
        return FlatSource(
            data=data,
            coordinates=coordinates.to(device),
            timedeltas=None if timedeltas is None else timedeltas.to(device),
            device=device,
            shard_sizes=flat_shard_sizes,
            batch_sizes=batch_sizes,
        )

    def unflatten(self, data: torch.Tensor, **kwargs) -> "TabularSource":
        node_counts = [sample.shape[self.layout.grid] for sample in self.data]
        row_counts = [count * self.ensemble_size for count in node_counts]
        row_starts = np.cumsum([0] + row_counts[:-1])

        new_data = []
        for sample_index, rows in enumerate(row_counts):
            chunk = data.narrow(0, int(row_starts[sample_index]), rows)
            if self.layout.ensemble is not None:
                chunk = chunk.unflatten(0, (self.ensemble_size, node_counts[sample_index]))
            new_data.append(chunk)

        return self.clone(data=new_data, **kwargs)

    def apply_pairwise(
        self,
        other: "TabularSource",
        func: Callable,
        *,
        per_sample_kwargs: dict[str, Sequence[Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply a loss function to this view and another view, returning the result."""
        if not isinstance(other, TabularSource):
            msg = f"Other source must be a TabularSource; got {type(other).__name__}."
            raise TypeError(msg)
        if self.layout != other.layout:
            msg = f"Both sources must have the same layout; got {self.layout!r} and {other.layout!r}."
            raise ValueError(msg)
        if len(self.data) != len(other.data):
            msg = f"Both sources must have the same number of samples; got {len(self.data)} and {len(other.data)}."
            raise ValueError(msg)
        # assert self.variables == other.variables, f"Both views must have the same variables; got {self.variables} and {other.variables}."

        per_sample_kwargs = {} if per_sample_kwargs is None else per_sample_kwargs
        if kwargs.keys() & per_sample_kwargs.keys():
            raise ValueError("Loss arguments cannot be both shared and per-sample.")
        for name, values in per_sample_kwargs.items():
            if len(values) != len(self.data):
                raise ValueError(f"Loss argument {name!r} requires one value per sample ({len(self.data)}).")

        losses = []
        non_empty = []
        for i, (pred, target) in enumerate(zip(self.data, other.data)):
            # every axis but the ensemble one must line up for this to work
            assert _shape_without_ensemble_dim(pred, self.layout) == _shape_without_ensemble_dim(target, self.layout), (
                f"Sample {i} of both views must have the same shape apart from the ensemble axis; "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}."
            )
            assert torch.all(
                self.coordinates[i] == other.coordinates[i]
            ), f"Sample {i} of both views must have the same coordinates; got {self.coordinates[i]} and {other.coordinates[i]}."
            sample_kwargs = kwargs | {name: values[i] for name, values in per_sample_kwargs.items()}

            losses.append(
                func(
                    pred,
                    target,
                    layout=self.layout,
                    statistics=self.statistics,
                    name_to_index=self.name_to_index,
                    **sample_kwargs,
                )
            )
            # Handle empty batches: a fully-empty worker returns a graph-connected 0
            non_empty.append(pred.shape[self.layout.grid] > 0)

        if not losses:
            msg = "Cannot apply a loss to an empty sparse source view."
            raise ValueError(msg)

        stacked = torch.stack(losses)
        num_non_empty = sum(non_empty)
        # Divide by the number of non-empty samples (>= 1) rather than the batch size.
        # When every sample is empty, the stacked tensor is all-zero and graph-connected,
        # so summing and dividing by 1 preserves the zero gradient path.
        return stacked.sum(dim=0) / max(num_non_empty, 1)

    def shard(self, group: ProcessGroup | None) -> "TabularSource":
        """Not supported: observation grids vary per sample."""
        if self.shard_sizes is not None or not model_is_distributed(group):
            return self

        msg = (
            f"Sharding is implemented for gridded sources only, but {self.name!r} is tabular. "
            "Tabular sources are sharded at read time by the reader instead."
        )
        raise NotImplementedError(msg)

    def allgather(self, group: ProcessGroup | None) -> "TabularSource":
        """Allgather this view across the given process group.

        This is a collective operation that synchronizes all processes in
        the group. The view's data and coordinates are allgathered across
        the grid dimension while metadata like layout and variables are
        unchanged.

        Parameters
        ----------
        group : ProcessGroup or None
            The process group to allgather across. If None, defaults to the
            global process group.

        Returns
        -------
        TabularSource
            A new view with allgathered data and coordinates.
        """
        if self.shard_sizes is None:
            return self  # nothing to gather

        # Validate every per-window descriptor up front, so a wrong-group gather is reported
        # before any partial sequence of per-window collectives has been issued.
        for sample_idx, sample_shard_sizes in enumerate(self.shard_sizes):
            for window_idx, window_shard_sizes in enumerate(sample_shard_sizes):
                check_shard_sizes_match_group(
                    window_shard_sizes,
                    group,
                    context=f"tabular source view {self.name!r} (sample {sample_idx}, window {window_idx})",
                )

        if not model_is_distributed(group):
            return self.clone(shard_sizes=None)

        gathered_data = []
        gathered_coords = []
        gathered_timedeltas = []
        gathered_boundaries = []
        for data, coords, timedeltas, boundaries, shard_sizes in zip(
            self.data, self.coordinates, self.timedeltas, self.boundaries, self.shard_sizes
        ):
            gathered_data.append([])
            gathered_coords.append([])
            gathered_timedeltas.append([])
            gathered_boundaries.append([])
            boundary_offset = 0

            # reconstruct per-window tensors using boundaries, then allgather and concatenates
            for window_slice, window_shard_sizes in zip(boundaries, shard_sizes):
                window_size = window_slice.stop - window_slice.start
                window_data = data.narrow(self.layout.grid, window_slice.start, window_size)
                gathered_window_data = gather_tensor(
                    window_data,
                    dim=self.layout.grid,
                    sizes=window_shard_sizes,
                    mgroup=group,
                )
                gathered_data[-1].append(gathered_window_data)

                # TODO(Jan): coordinates/td/boundaries is None?
                window_coords = coords[window_slice]
                gathered_window_coords = gather_tensor(
                    window_coords,
                    dim=0,
                    sizes=window_shard_sizes,
                    mgroup=group,
                )
                gathered_coords[-1].append(gathered_window_coords)

                window_timedeltas = timedeltas[window_slice]
                gathered_window_timedeltas = gather_tensor(
                    window_timedeltas,
                    dim=0,
                    sizes=window_shard_sizes,
                    mgroup=group,
                )
                gathered_timedeltas[-1].append(gathered_window_timedeltas)

                new_window_size = sum(window_shard_sizes)
                gathered_boundaries[-1].append(slice(boundary_offset, boundary_offset + new_window_size))
                boundary_offset += new_window_size

            gathered_data[-1] = torch.cat(gathered_data[-1], dim=self.layout.grid)
            gathered_coords[-1] = torch.cat(gathered_coords[-1], dim=0)
            gathered_timedeltas[-1] = torch.cat(gathered_timedeltas[-1], dim=0)

        return self.clone(
            data=gathered_data,
            coordinates=gathered_coords,
            timedeltas=gathered_timedeltas,
            boundaries=gathered_boundaries,
            shard_sizes=None,
        )

    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "TabularSource":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        new_data = [self._index_vars(t, indices) for t in self.data]
        return self.clone(data=new_data, spec=self.spec.select_variables(indices))

    def select_time(self, indices: "slice | Sequence[int] | int") -> "TabularSource":
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
        TabularSource
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
        new_shard_sizes = [] if self.shard_sizes is not None else None

        for sample_idx, sample_bounds in enumerate(self.boundaries):
            selected_slices = [sample_bounds[t] for t in idx_list]
            sample_data = self.data[sample_idx]
            data_pieces = [sample_data.narrow(self.layout.grid, s.start, s.stop - s.start) for s in selected_slices]
            new_data.append(
                torch.cat(data_pieces, dim=self.layout.grid)
                if data_pieces
                else sample_data.narrow(self.layout.grid, 0, 0)
            )

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

            if new_shard_sizes is not None and self.shard_sizes is not None:
                sample_shard_sizes = self.shard_sizes[sample_idx]
                new_shard_sizes.append([sample_shard_sizes[t] for t in idx_list])

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
            shard_sizes=new_shard_sizes,
        )

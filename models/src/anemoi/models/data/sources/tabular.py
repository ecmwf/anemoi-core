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
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import torch
from rich.tree import Tree
from torch.distributed import ProcessGroup

from anemoi.models.data.flat import FlatSource
from anemoi.models.data.sources.base import Source
from anemoi.models.data.sources.base import Template
from anemoi.models.data.sources.base import _index_list
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import check_shard_sizes_match_group
from anemoi.models.distributed.utils import model_is_distributed

LOGGER = logging.getLogger(__name__)


def _sample_condition(sample: torch.Tensor, condition: torch.Tensor, sample_index: int) -> torch.Tensor:
    """Select one sample's slice of a ``(batch, 1, ensemble, 1, 1)`` condition, shaped to broadcast against it."""
    if condition.ndim != 5:
        msg = f"Expected transport condition to be 5D, got shape {tuple(condition.shape)}."
        raise ValueError(msg)

    sample_condition = condition[sample_index, 0, :, 0, 0]
    if sample.ndim <= 2:
        if sample_condition.numel() != 1:
            msg = "Sparse observation data without an ensemble axis requires ensemble_size == 1."
            raise NotImplementedError(msg)
        return sample_condition.reshape(())

    view_shape = [sample_condition.shape[0]] + [1] * (sample.ndim - 1)
    return sample_condition.reshape(view_shape)


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


# No slots=True: it rebuilds the class, which breaks the zero-argument super() in __post_init__.
@dataclass(frozen=True, eq=False, kw_only=True)
class TabularSource(Source):
    """Tabular data source: points that change from sample to sample (e.g. observations).

    Every payload field is a list with one entry per sample, since the number of
    points differs between samples.

    Parameters
    ----------
    data : list[torch.Tensor]
        One ``(ensemble, grid_i, variables)`` tensor per sample, laid out per ``layout``.
    coordinates : list[torch.Tensor]
        One ``(grid_i, 2)`` tensor of latitudes and longitudes in radians per sample.
    timedeltas : list[torch.Tensor]
        One ``(grid_i,)`` tensor of per-point time offsets per sample.
    boundaries : list[tuple[slice, ...]]
        The time windows of each sample, as slices along the grid axis.
    shard_sizes : list[list[ShardSizes]], optional
        Per sample and per time window, the per-rank point counts from read-time
        sharding. ``None`` when the source is replicated.
    """

    data: list[torch.Tensor]
    coordinates: list[torch.Tensor]
    timedeltas: list[torch.Tensor]
    boundaries: list[tuple[slice, ...]]
    shard_sizes: list[list[ShardSizes] | None] | None = None

    _PAYLOAD_FIELDS = ("data", "timedeltas")

    @property
    def coordinates_are_static(self) -> bool:
        """Always ``False``: tabular points change from sample to sample."""
        return False

    def __post_init__(self):
        super().__post_init__()
        for field_name in ("timedeltas", "boundaries"):
            if getattr(self, field_name) is None:
                msg = f"{self.__class__.__name__} {self.name!r} requires {field_name}; got None."
                raise ValueError(msg)

        ts = tuple(len(t) for t in self.timedeltas)
        cs = tuple(len(c) for c in self.coordinates)
        if ts != cs:
            msg = (
                f"{self.__class__.__name__} {self.name!r} timedeltas and coordinates must contain the same number of nodes, "
                f"got {sum(ts)} and {sum(cs)}."
            )
            raise ValueError(msg)

        if self.layout.has_axis("time") or self.layout.has_axis("batch"):
            msg = (
                f"{self.__class__.__name__} requires a layout without time and batch axes; the time windows are "
                f"given by 'boundaries' and the batch is a list. Got {self.layout!r}."
            )
            raise ValueError(msg)

        if isinstance(self.data, torch.Tensor | np.ndarray):
            msg = f"{self.__class__.__name__} data must be a list of tensors, not a single tensor."
            raise TypeError(msg)

        if self.data is not None:
            layout = self.layout.normalized(self.layout.ndim)
            for sample, sample_coords in zip(self.data, self.coordinates, strict=True):
                if tuple(sample_coords.shape) != (sample.shape[self.layout.grid], 2):
                    raise ValueError(f"Source {self.name!r} requires one latitude/longitude pair per node.")

                num_channels = sample.shape[layout.variables]
                if num_channels != len(self.variables):
                    raise ValueError(
                        f"{self.__class__.__name__} {self.name!r} has {num_channels} variable channels "
                        f"but {len(self.variables)} names."
                    )

        if self.data and any(sample.dtype != self.data[0].dtype for sample in self.data):
            raise ValueError(f"{self.__class__.__name__} {self.name!r} requires the same dtype for every sample.")

    @property
    def grid_size(self) -> None:
        """Full grid size before sharding.

        In tabular sources, the concept of a full grid size does not apply, hence it returns ``None``.
        """
        return None

    @property
    def batch_size(self) -> int:
        """Number of samples (batch size) in this source."""
        return len(self.data)

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members per sample, 1 when the layout has no ensemble axis."""
        if self.layout.ensemble is None:
            return 1

        ensemble_size = [data.shape[self.layout.ensemble] for data in self.data]

        if len(set(ensemble_size)) != 1:
            msg = f"Inconsistent ensemble sizes across batch samples: {ensemble_size}"
            raise ValueError(msg)

        return ensemble_size[0]

    @property
    def time_size(self) -> int:
        """Number of time windows in this source, taken from ``boundaries``."""
        time_sizes = [len(boundaries) for boundaries in self.boundaries]
        if len(set(time_sizes)) != 1:
            msg = f"Inconsistent time sizes across batch samples: {time_sizes}"
            raise ValueError(msg)

        return time_sizes[0]

    @property
    def device(self) -> torch.device:
        """Device of the source's data tensor."""
        return self.data[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the source's data tensor."""
        return self.data[0].dtype

    def template(self) -> "TabularTemplate":
        """Return this source without its data (see :class:`TabularTemplate`)."""
        return TabularTemplate(
            **self._metadata_kwargs(),
            coordinates=self.coordinates,
            timedeltas=self.timedeltas,
            boundaries=self.boundaries,
            shard_sizes=self.shard_sizes,
            ensemble_size=self.ensemble_size,
        )

    def map_data(self, func: Callable[[torch.Tensor], torch.Tensor], **overrides) -> "TabularSource":
        """Return a new view with ``func`` applied to each data tensor.

        For plain tensor operations (``.to(dtype)``, ``.detach()``, ``.cpu()``, ...): ``func``
        takes only the tensor, and the data is not cloned first. It is applied once to a
        gridded source and once per sample to a tabular one. Use :meth:`apply_func` for
        functions that need the source's statistics or variable indices, such as processors.

        ``func`` must not modify its input in place; return a new tensor instead.
        """
        new_data = [func(t) for t in self.data]
        return self.clone(data=new_data, **overrides)

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

    @property
    def is_tabular(self) -> bool:
        return True

    @property
    def grid_shard_sizes(self) -> ShardSizes:
        # Sharded per time window, not along a single grid axis.
        return None
    
    @property
    def is_sharded(self) -> bool:
        """Return ``True`` if the source is sharded per time window."""
        return self.shard_sizes is not None and any(sizes is not None for sizes in self.shard_sizes)

    @property
    def condition_shape(self) -> tuple[int, int, int, int, int]:
        if not self.data:
            msg = "Cannot infer condition shape from an empty sparse data list."
            raise ValueError(msg)
        return self.batch_size, 1, self.ensemble_size, 1, 1

    def _check_same_samples(self, others: Sequence[Source]) -> None:
        self._check_same_structure(others)
        for other in others:
            if len(other.data) != len(self.data):
                msg = (
                    "Sparse transport data lists must have the same length, "
                    f"got {len(self.data)} and {len(other.data)}."
                )
                raise ValueError(msg)

    def _check_condition_batch(self, condition: torch.Tensor) -> None:
        if condition.shape[0] != len(self.data):
            msg = f"Condition batch size {condition.shape[0]} does not match sparse data length {len(self.data)}."
            raise ValueError(msg)

    def zip_map_data(self, fn: Callable[..., torch.Tensor], *others: Source) -> "TabularSource":
        self._check_same_samples(others)
        samples = zip(self.data, *(other.data for other in others), strict=True)
        return self.clone(data=[fn(*sample_group) for sample_group in samples])

    def map_with_condition(
        self,
        fn: Callable[..., torch.Tensor],
        condition: torch.Tensor,
        *others: Source,
    ) -> "TabularSource":
        self._check_same_samples(others)
        self._check_condition_batch(condition)
        samples = zip(self.data, *(other.data for other in others), strict=True)
        return self.clone(
            data=[
                fn(*sample_group, _sample_condition(sample_group[0], condition, index))
                for index, sample_group in enumerate(samples)
            ],
        )

    def condition_per_sample(self, condition: torch.Tensor) -> list[torch.Tensor]:
        self._check_condition_batch(condition)
        return [_sample_condition(sample, condition, index) for index, sample in enumerate(self.data)]

    def randn_like(self, model_comm_group: ProcessGroup | None = None) -> "TabularSource":
        del model_comm_group  # each sample is drawn independently; windows are not grid-sharded
        # torch.randn (not randn_like), as the gridded path draws through randn_with_grid_sharding.
        return self.clone(
            data=[torch.randn(sample.shape, dtype=sample.dtype, device=sample.device) for sample in self.data],
        )

    def pairwise(self, other: Source, func: Callable[..., torch.Tensor], *args, **kwargs) -> torch.Tensor:
        if not isinstance(other, TabularSource):
            raise TypeError(f"Other source must be a TabularSource; got {type(other).__name__}.")

        if self.layout != other.layout:
            raise ValueError(f"Both sources must have the same layout; got {self.layout!r} and {other.layout!r}.")

        if len(self.data) != len(other.data):
            raise ValueError(
                f"Both sources must have the same number of samples; got {len(self.data)} and {len(other.data)}."
            )

        per_sample_kwargs = kwargs.pop("per_sample_kwargs", None) or {}
        if kwargs.keys() & per_sample_kwargs.keys():
            raise ValueError("Loss arguments cannot be both shared and per-sample.")

        for name, values in per_sample_kwargs.items():
            if len(values) != len(self.data):
                raise ValueError(f"Loss argument {name!r} requires one value per sample ({len(self.data)}).")

        losses = []
        non_empty = []
        for i, (pred_sample, target_sample) in enumerate(zip(self.data, other.data)):
            # Every axis but the ensemble one must line up for this to work.
            if self.layout.ensemble is None:
                pred_shape = tuple(pred_sample.shape)
                target_shape = tuple(target_sample.shape)
            else:
                ensemble_axis = self.layout.axis("ensemble", ndim=pred_sample.ndim)
                pred_shape = tuple(size for dim, size in enumerate(pred_sample.shape) if dim != ensemble_axis)
                target_shape = tuple(size for dim, size in enumerate(target_sample.shape) if dim != ensemble_axis)

            assert pred_shape == target_shape, (
                f"Sample {i} of both views must have the same shape apart from the ensemble axis; "
                f"got {tuple(pred_sample.shape)} and {tuple(target_sample.shape)}."
            )
            assert torch.equal(self.coordinates[i], other.coordinates[i]), (
                f"Sample {i} of both views must have the same coordinates; "
                f"got {self.coordinates[i]} and {other.coordinates[i]}."
            )
            sample_kwargs = kwargs | {name: values[i] for name, values in per_sample_kwargs.items()}
            losses.append(
                func(
                    pred_sample,
                    target_sample,
                    *args,
                    layout=self.layout,
                    statistics=self.statistics,
                    name_to_index=self.name_to_index,
                    **sample_kwargs,
                ),
            )
            # A fully-empty worker contributes a graph-connected zero without
            # reducing the mean for non-empty workers.
            non_empty.append(pred_sample.shape[self.layout.grid] > 0)

        if not losses:
            raise ValueError("Cannot apply a loss to an empty sparse source view.")

        stacked = torch.stack(losses)
        return stacked.sum(dim=0) / max(sum(non_empty), 1)

    def flatten(self) -> FlatSource:
        """Flatten the source into rows, one per ``(sample, member, point)``.

        Each sample's ensemble axis is folded into its point axis, and the samples are joined.
        """
        folded = [_fold_members(self, sample) for sample in self.data]
        data = torch.cat(folded, dim=0) if len(folded) > 1 else folded[0]
        return replace(self.template().flatten().to(data.device), data=data)

    def shard(self, group: ProcessGroup | None) -> "TabularSource":
        """Not supported: observation grids vary per sample."""
        if self.is_sharded or not model_is_distributed(group):
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
        if self.shard_sizes is None or all(sizes is None for sizes in self.shard_sizes):
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
        return self.clone(data=new_data, **self._select_variable_metadata(indices))

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
        idx_list = _index_list(indices, self.time_size)

        new_data = []
        new_coords = []
        new_timedeltas = []
        new_boundaries = []
        new_shard_sizes = [] if self.is_sharded else None

        for sample_idx, sample_bounds in enumerate(self.boundaries):
            selected_slices = [sample_bounds[t] for t in idx_list]
            sample_data = self.data[sample_idx]
            data_pieces = [sample_data.narrow(self.layout.grid, s.start, s.stop - s.start) for s in selected_slices]
            new_data.append(
                torch.cat(data_pieces, dim=self.layout.grid)
                if data_pieces
                else sample_data.narrow(self.layout.grid, 0, 0)
            )

            sample_coords = self.coordinates[sample_idx]
            coord_pieces = [sample_coords[s.start : s.stop] for s in selected_slices]
            new_coords.append(torch.cat(coord_pieces, dim=0) if coord_pieces else sample_coords[:0])

            sample_td = self.timedeltas[sample_idx]
            td_pieces = [sample_td[s.start : s.stop] for s in selected_slices]
            new_timedeltas.append(torch.cat(td_pieces, dim=0) if td_pieces else sample_td[:0])

            if new_shard_sizes is not None:
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
            coordinates=new_coords,
            timedeltas=new_timedeltas,
            boundaries=new_boundaries,
            shard_sizes=new_shard_sizes,
        )

    def concat_time(self, *others: "TabularSource") -> "TabularSource":
        """Return a new view with the time windows of ``others`` appended after this source's.

        Sample by sample, data, coordinates and timedeltas are concatenated along the node axis,
        and the window boundaries and shard sizes are extended accordingly.

        Parameters
        ----------
        *others : TabularSource
            Sources with the same samples, variables, layout and ensemble size.

        Returns
        -------
        TabularSource
            A new view with ``self.time_size + sum(other.time_size)`` windows.
        """
        sources = (self, *others)
        for other in others:
            if not isinstance(other, TabularSource):
                msg = f"Cannot append {type(other).__name__} windows to TabularSource {self.name!r}."
                raise TypeError(msg)
            mismatch = {
                "batch size": (self.batch_size, other.batch_size),
                "variables": (tuple(self.variables), tuple(other.variables)),
                "layout": (self.layout, other.layout),
                "ensemble size": (self.ensemble_size, other.ensemble_size),
                "sharding": (self.shard_sizes is None, other.shard_sizes is None),
            }
            mismatch = {key: values for key, values in mismatch.items() if values[0] != values[1]}
            if mismatch:
                msg = f"Cannot append the windows of source {other.name!r} to {self.name!r}: mismatched {mismatch}."
                raise ValueError(msg)

        grid_axis = self.layout.grid
        new_data, new_coords, new_timedeltas, new_boundaries = [], [], [], []
        new_shard_sizes = None if self.shard_sizes is None else []
        for sample_idx in range(self.batch_size):
            new_data.append(torch.cat([source.data[sample_idx] for source in sources], dim=grid_axis))
            new_coords.append(torch.cat([source.coordinates[sample_idx] for source in sources], dim=0))
            new_timedeltas.append(torch.cat([source.timedeltas[sample_idx] for source in sources], dim=0))

            offset = 0
            boundaries = []
            for source in sources:
                for window in source.boundaries[sample_idx]:
                    boundaries.append(slice(window.start + offset, window.stop + offset))
                offset += len(source.coordinates[sample_idx])
            new_boundaries.append(tuple(boundaries))

            if new_shard_sizes is not None:
                new_shard_sizes.append([sizes for source in sources for sizes in source.shard_sizes[sample_idx]])

        return self.clone(
            data=new_data,
            coordinates=new_coords,
            timedeltas=new_timedeltas,
            boundaries=new_boundaries,
            shard_sizes=new_shard_sizes,
        )

    def tree(self, prefix: str = "") -> Tree:
        """Return a tree representation of the tabular source.

        Example
        -------
        >>> source = TabularSource(...)
        >>> tree = source.tree()
        >>> print(tree)
        era5 | TabularSource[torch.float32, cuda:0]
            Dim 0 (batch): 3
            Dim 1 (grid): 3 time slices
                Sample I: 42312 <- 12312 + 10000 + 20000
                Sample II: 49394 <- 20000 + 15000 + 14394
                Sample III: 12319 <- 5000 + 4000 + 3319
            Dim 2 (ensemble): 1
            Dim 3 (variables): 83
        """
        dims = {getattr(self.layout, name): name for name in self.layout.AXES if getattr(self.layout, name) is not None}

        dtype = self.data[0].dtype
        device = self.data[0].device
        tree = Tree(prefix + self.name + " | " + self.__class__.__name__ + f"[{dtype}, {device}]")
        tree.add(f"Dim 0 (batch): {len(self.data)}")
        for axis in range(self.data[0].ndim):
            name = dims[axis]
            if name == "grid":
                tree.add(f"Dim {axis+1} ({name}): {self.time_size} time slices")
                for i, sample in enumerate(self.data):
                    time_slice_lengths = [str(s.stop - s.start) for s in self.boundaries[i]]
                    tree.add(f"\tSample {i+1}: {sample.shape[axis]} <- {' + '.join(time_slice_lengths)}")
            else:
                tree.add(f"Dim {axis+1} ({name}): {self.data[0].shape[axis]}")

        return tree


@dataclass(frozen=True, eq=False, kw_only=True)
class TabularTemplate(Template):
    """A :class:`TabularSource` without its data.

    Parameters
    ----------
    coordinates : list[torch.Tensor]
        One ``(grid_i, 2)`` tensor of latitudes and longitudes in radians per sample.
    timedeltas : list[torch.Tensor]
        One ``(grid_i,)`` tensor of per-point time offsets per sample.
    boundaries : list[tuple[slice, ...]]
        The time windows of each sample, as slices along the grid axis.
    shard_sizes : list[list[ShardSizes]], optional
        Per sample and per time window, the per-rank point counts. ``None`` when replicated.
    ensemble_size : int
        Number of ensemble members per sample.
    """

    coordinates: list[torch.Tensor]
    timedeltas: list[torch.Tensor]
    boundaries: list[tuple[slice, ...]]
    shard_sizes: list[list[ShardSizes]] | None = None
    ensemble_size: int

    @property
    def coordinates_are_static(self) -> bool:
        """Always ``False``: tabular points change from sample to sample."""
        return False

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return len(self.coordinates)

    @property
    def time_size(self) -> int:
        """Number of time windows, taken from ``boundaries``."""
        time_sizes = {len(boundaries) for boundaries in self.boundaries}
        if len(time_sizes) != 1:
            msg = f"Inconsistent time sizes across batch samples: {sorted(time_sizes)}"
            raise ValueError(msg)
        return time_sizes.pop()

    @property
    def node_counts(self) -> list[int]:
        """Number of points in each sample."""
        return [coords.shape[0] for coords in self.coordinates]

    def flatten(self) -> FlatSource:
        """Return the flat nodes: every sample's points repeated for each member, then joined."""
        # coordinates and timedeltas are repeated per member to line up with the folded data
        repeated_coords = [c.repeat(self.ensemble_size, 1) for c in self.coordinates]
        repeated_timedeltas = [td.repeat(self.ensemble_size) for td in self.timedeltas]
        coordinates = torch.cat(repeated_coords, dim=0) if len(repeated_coords) > 1 else repeated_coords[0]
        timedeltas = torch.cat(repeated_timedeltas, dim=0) if len(repeated_timedeltas) > 1 else repeated_timedeltas[0]

        # Flatten per-window shard sizes into one list for the concatenated data.
        # NOTE this changes the order of observations when gathering:
        #   GPU0  GPU1   GPU0  GPU1            GPU0        GPU1
        #   w1_0, w1_1 | w2_0, w2_1  becomes  w1_0, w2_0, w1_1, w2_1
        flat_shard_sizes = None
        if self.shard_sizes is not None and all(sizes is not None for sizes in self.shard_sizes):
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

        # moving grids need one graph per (sample, member); see DynamicGraphProvider
        batch_sizes = tuple(count for count in self.node_counts for _ in range(self.ensemble_size))

        return FlatSource(
            data=None,
            coordinates=coordinates,
            timedeltas=timedeltas.to(coordinates.device),
            device=coordinates.device,
            shard_sizes=flat_shard_sizes,
            batch_sizes=batch_sizes,
        )

    def unflatten(self, data: torch.Tensor) -> TabularSource:
        """Build a :class:`TabularSource` from rows laid out like :meth:`flatten`."""
        row_counts = [count * self.ensemble_size for count in self.node_counts]
        expected = (sum(row_counts), self.n_variables)
        if tuple(data.shape) != expected:
            msg = (
                f"Template {self.name!r} expects flat data of shape {expected} "
                f"(points over all samples and members, variables), got {tuple(data.shape)}."
            )
            raise ValueError(msg)

        row_starts = np.cumsum([0] + row_counts[:-1])
        samples = []
        for sample_index, rows in enumerate(row_counts):
            chunk = data.narrow(0, int(row_starts[sample_index]), rows)
            if self.layout.ensemble is not None:
                chunk = chunk.unflatten(0, (self.ensemble_size, self.node_counts[sample_index]))
            samples.append(chunk)

        return TabularSource(
            **self._metadata_kwargs(),
            data=samples,
            coordinates=self.coordinates,
            timedeltas=self.timedeltas,
            boundaries=self.boundaries,
            shard_sizes=self.shard_sizes,
        )

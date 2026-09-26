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

import einops
import torch
from rich.tree import Tree
from torch.distributed import ProcessGroup

from anemoi.models.data.flat import FlatSource
from anemoi.models.data.sources import FLATTEN_PATTERN
from anemoi.models.data.sources.base import Source
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import check_shard_sizes_match_group
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed

LOGGER = logging.getLogger(__name__)


class GriddedSource(Source):
    """Gridded data source."""

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a single tensor, not a list."
            raise TypeError(msg)

        if isinstance(self.coordinates, list):
            msg = f"Source {self.name!r} coordinates must be a tensor, not a list."
            raise TypeError(msg)

        if self.layout.time is None:
            msg = f"{self.__class__.__name__} requires a layout with a time axis; got {self.layout!r}."
            raise ValueError(msg)

        if self.data is not None:
            # If data is provided, check that the number of channels matches the number of variable names.
            num_channels = self.data.shape[self.layout.variables]
            if num_channels != len(self.variables):
                raise ValueError(
                    f"{self.__class__.__name__} {self.name!r} has {num_channels} variable channels "
                    f"but {len(self.variables)} names."
                )

    @property
    def device(self) -> torch.device:
        """Device of the source's data tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the source's data tensor."""
        return self.data.dtype

    @property
    def grid_size(self) -> int:
        """Full grid size before sharding; ``None`` for observation datasets."""
        return self.data.shape[self.layout.grid]

    @property
    def batch_size(self) -> int:
        """Number of samples (batch size) in this source."""
        if self.layout.batch is None:
            raise ValueError(f"{self.__class__.__name__}.batch_size requires a layout with a batch axis.")

        return self.data.shape[self.layout.batch]

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members per sample, 1 when the layout has no ensemble axis."""
        if self.layout.ensemble is None:
            return 1

        return self.data.shape[self.layout.ensemble]

    @property
    def time_size(self) -> int:
        """Number of time steps in this source."""
        if self.layout.time is None:
            raise ValueError(f"{self.__class__.__name__}.time_size requires a layout with a time axis.")

        return self.data.shape[self.layout.time]

    def empty(self) -> "EmptyGriddedSource":
        """Return a copy with ``data`` dropped, keeping shape metadata that ``data`` would otherwise supply.

        ``EmptyGriddedSource`` reads ``device``, ``dtype``, ``grid_size``, ``batch_size``
        and ``ensemble_size`` from values captured here, since none of them can be
        derived from a ``None`` tensor.
        """
        return EmptyGriddedSource(
            spec=self.spec,
            data=None,
            coordinates=self.coordinates,
            shard_sizes=self.shard_sizes,
            _device=self.device,
            _dtype=self.dtype,
            _grid_size=self.grid_size,
            _batch_size=self.batch_size,
            _ensemble_size=self.ensemble_size,
        )

    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "GriddedSource":
        """Apply a function to this view, returning a new view with the same metadata."""
        new_data = func(
            self.data if in_place else self.data.clone(),
            statistics=self.statistics,
            name_to_index=self.name_to_index,
            **kwargs,
        )
        return self.clone(data=new_data)

    def flatten(self) -> "GriddedSource":
        """Flatten the gridded source into a flat source."""
        assert (
            self.layout.batch is not None
        ), f"{self.__class__.__name__} requires to have a batch axis to be flattened."

        if self.data is not None:
            current_pattern = self.layout.normalized(self.data.ndim).pattern
            flattened_data = einops.rearrange(self.data, f"{current_pattern} -> {FLATTEN_PATTERN}")
        else:
            flattened_data = None

        if self.coordinates is None:
            raise ValueError(f"{self.__class__.__name__} {self.name!r} requires coordinates to be flattened.")

        # static grids share one (grid, 2) coordinate set; moving grids carry one per sample, (batch, grid, 2)
        grid_size = self.coordinates.shape[-2]
        if self.coordinates.ndim == 2:
            expected_shape = (grid_size, 2)
            coords_pattern = "grid latlon -> (batch ensemble grid) latlon"
        elif self.coordinates.ndim == 3:
            expected_shape = (self.batch_size, grid_size, 2)
            coords_pattern = "batch grid latlon -> (batch ensemble grid) latlon"
        else:
            raise ValueError(
                f"{self.__class__.__name__} {self.name!r} coordinates must have shape (grid, 2) "
                f"or (batch, grid, 2), got {tuple(self.coordinates.shape)}."
            )
        if tuple(self.coordinates.shape) != expected_shape:
            raise ValueError(
                f"{self.__class__.__name__} {self.name!r} coordinates must have shape {expected_shape}, "
                f"got {tuple(self.coordinates.shape)}."
            )

        flattened_coords = einops.repeat(
            self.coordinates,
            coords_pattern,
            batch=self.batch_size,
            ensemble=self.ensemble_size,
        )
        # already on device; see Batch.to()

        return FlatSource(
            data=flattened_data,
            coordinates=flattened_coords,
            shard_sizes=self.shard_sizes,
            device=self.device,
            # moving grids need one graph per (sample, member); see DynamicGraphProvider
            batch_sizes=(
                None if self.coordinates_are_static else (grid_size,) * (self.batch_size * self.ensemble_size)
            ),
        )

    def unflatten(self, data: torch.Tensor, **kwargs) -> "GriddedSource":
        new_data = einops.rearrange(
            data,
            f"{FLATTEN_PATTERN} -> {self.layout.normalized(self.data.ndim).pattern}",
            batch=self.batch_size,
            ensemble=self.ensemble_size,
            time=self.data.shape[self.layout.time],
        )

        return self.clone(data=new_data, **kwargs)

    def shard(self, group: ProcessGroup | None) -> "GriddedSource":
        """Split this source across ``group`` along its grid axis."""
        if self.shard_sizes is not None:
            return self
        if not model_is_distributed(group):
            return self

        grid_dim = self.layout.axis("grid", ndim=self.data.ndim)
        sizes = get_shard_sizes(self.data, grid_dim, model_comm_group=group)
        coordinates = self.coordinates
        if coordinates is not None:
            coordinates = shard_tensor(coordinates, -2, sizes, group)

        return self.clone(
            data=shard_tensor(self.data, grid_dim, sizes, group),
            coordinates=coordinates,
            shard_sizes=sizes,
        )

    def allgather(self, group: ProcessGroup | None) -> "GriddedSource":
        """Allgather this view across the given process group.

        This is a collective operation that synchronizes all processes in
        the group. The view's data is allgathered across the grid dimension
        while metadata like layout and variables are unchanged.

        No-op when the view is replicated (shard_sizes is None) or when group spans
        a single rank, which makes repeated calls idempotent.

        Coordinates are gathered alongside the data and so must already be on the same
        device; :meth:`Batch.to <anemoi.models.data.batch.Batch.to>` guarantees that.
        A mismatch raises ValueError rather than being silently transferred.

        Parameters
        ----------
        group : ProcessGroup or None
            The process group to allgather across. None means single-rank, i.e. the
            view is already complete.

        Returns
        -------
        GriddedSource
            A new view with allgathered data, or self when already full-grid.

        Raises
        ------
        ValueError
            If shard_sizes does not describe ``group``, or if coordinates and data
            live on different devices.
        """
        if self.shard_sizes is None:
            return self  # replicated: nothing to gather

        # Validate before the single-rank fast path below, so that gathering over the wrong
        # process group is reported instead of silently dropping the shard metadata.
        check_shard_sizes_match_group(self.shard_sizes, group, context=f"gridded source view {self.name!r}")

        if not model_is_distributed(group):
            return self.clone(shard_sizes=None)

        gathered_coords = self.coordinates
        if gathered_coords is not None and gathered_coords.device != self.device:
            # Batch.to() is responsible for putting coordinates on the model device;
            # a mismatch here would be gathered on the wrong backend, so say so plainly.
            msg = (
                f"Gridded source view {self.name!r} has coordinates on {gathered_coords.device} "
                f"but data on {self.device}; both must be on the same device before allgather. "
                "Batch.to() moves them together - check that this batch went through it."
            )
            raise ValueError(msg)

        gathered_data = gather_tensor(
            self.data,
            dim=self.layout.grid,
            sizes=self.shard_sizes,
            mgroup=group,
        )
        if gathered_coords is not None:
            gathered_coords = gather_tensor(
                gathered_coords,
                dim=-2,
                sizes=self.shard_sizes,
                mgroup=group,
            )

        return self.clone(data=gathered_data, coordinates=gathered_coords, shard_sizes=None)

    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "GriddedSource":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        new_data = self._index_vars(self.data, indices)
        return self.clone(data=new_data, spec=self.spec.select_variables(indices))

    def select_time(self, indices: "slice | Sequence[int] | int") -> "GriddedSource":
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
        GriddedSource
            A new view with the same :class:`TensorLayout` but reduced
            time extent.
        """
        if self.layout.time is None:
            msg = f"Layout {self.layout!r} has no time axis."
            raise ValueError(msg)

        if isinstance(indices, slice):
            time_size = self.data.shape[self.layout.time]
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

    def tree(self, prefix: str = "") -> Tree:
        """Return a tree representation of the gridded source.

        Example
        -------
        >>> source = GriddedSource(...)
        >>> tree = source.tree()
        >>> print(tree)
        era5 | GriddedSource[torch.float32, cuda:0]
            Dim 0 (time): 3
            Dim 1 (grid): 40980
            Dim 2 (ensemble): 1
            Dim 3 (variables): 83
        """
        dims = {getattr(self.layout, name): name for name in self.layout.AXES if getattr(self.layout, name) is not None}

        tree = Tree(prefix + self.name + " | " + self.__class__.__name__ + f"[{self.data.dtype}, {self.data.device}]")
        for axis in range(self.data.ndim):
            tree.add(f"Dim {axis} ({dims[axis]}): {self.data.shape[axis]}")

        if self.shard_sizes is not None:
            tree.add(f"Shard sizes: {self.shard_sizes}")

        return tree


@dataclass(frozen=True)
class EmptyGriddedSource(GriddedSource):
    """A :class:`GriddedSource` with no data, produced by :meth:`GriddedSource.empty`.

    ``device``, ``dtype``, ``grid_size``, ``batch_size`` and ``ensemble_size`` are
    normally read off ``self.data``; with ``data=None`` that is no longer possible,
    so this subclass carries them as explicit fields instead and overrides the
    properties to return them.
    """

    _device: torch.device = None
    _dtype: torch.dtype = None
    _grid_size: int | None = None
    _batch_size: int = 0
    _ensemble_size: int = 1

    @property
    def device(self) -> torch.device:
        """Device the source lived on before its data was dropped."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type the source had before its data was dropped."""
        return self._dtype

    @property
    def grid_size(self) -> int | None:
        """Full grid size before sharding, captured before data was dropped."""
        return self._grid_size

    @property
    def batch_size(self) -> int:
        """Number of samples (batch size), captured before data was dropped."""
        return self._batch_size

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members, captured before data was dropped."""
        return self._ensemble_size

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
from anemoi.models.data.sources.base import Source
from anemoi.models.data.sources.base import _index_list
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import check_shard_sizes_match_group
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed

LOGGER = logging.getLogger(__name__)


# No slots=True: it rebuilds the class, which breaks the zero-argument super() in __post_init__.
@dataclass(frozen=True, kw_only=True)
class GriddedSource(Source):
    """Gridded data source: every sample of the batch shares one grid.

    Parameters
    ----------
    data : torch.Tensor or None
        ``(batch, time, ensemble, grid, variables)``, laid out per ``layout``.
        ``None`` only for :class:`EmptyGriddedSource`.
    coordinates : torch.Tensor
        ``(grid, 2)`` latitudes and longitudes in radians, shared by the whole batch.
    shard_sizes : ShardSizes, optional
        Per-rank grid sizes when the source is sharded, ``None`` when it is replicated.
    """

    data: torch.Tensor | None
    coordinates: torch.Tensor
    shard_sizes: ShardSizes | None = None

    @property
    def coordinates_are_static(self) -> bool:
        """Always ``True``: a gridded dataset keeps one grid for the whole run.

        The coordinate tensor is therefore shared by reference across batches rather
        than transferred each time (see :meth:`Source.to`).
        """
        return True

    # How a source's axes collapse into ``(nodes, features)``.
    FLATTEN_PATTERN = "(batch ensemble grid) (time variables)"

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a single tensor, not a list."
            raise TypeError(msg)

        if isinstance(self.coordinates, list):
            msg = f"Source {self.name!r} coordinates must be a tensor, not a list."
            raise TypeError(msg)

        if self.coordinates.ndim != 2:
            msg = f"Source {self.name!r} coordinates must have shape (grid, 2); got {tuple(self.coordinates.shape)}."
            raise ValueError(msg)

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
            **self._metadata_kwargs(),
            data=None,
            coordinates=self.coordinates,
            shard_sizes=self.shard_sizes,
            _device=self.device,
            _dtype=self.dtype,
            _grid_size=self.grid_size,
            _batch_size=self.batch_size,
            _ensemble_size=self.ensemble_size,
        )

    def map_data(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "GriddedSource":
        """Return a new view with ``func`` applied to each data tensor.

        For plain tensor operations (``.to(dtype)``, ``.detach()``, ``.cpu()``, ...): ``func``
        takes only the tensor, and the data is not cloned first. It is applied once to a
        gridded source and once per sample to a tabular one. Use :meth:`apply_func` for
        functions that need the source's statistics or variable indices, such as processors.

        ``func`` must not modify its input in place; return a new tensor instead.
        """
        new_data = func(self.data)
        return self.clone(data=new_data)

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
            flattened_data = einops.rearrange(self.data, f"{current_pattern} -> {GriddedSource.FLATTEN_PATTERN}")
        else:
            flattened_data = None

        flattened_coords = einops.repeat(
            self.coordinates,
            "grid latlon -> (batch ensemble grid) latlon",
            batch=self.batch_size,
            ensemble=self.ensemble_size,
        )
        # already on device; see Batch.to()

        return FlatSource(
            data=flattened_data, coordinates=flattened_coords, shard_sizes=self.shard_sizes, device=self.device
        )

    def unflatten(self, data: torch.Tensor, **kwargs) -> "GriddedSource":
        new_data = einops.rearrange(
            data,
            f"{GriddedSource.FLATTEN_PATTERN} -> {self.layout.normalized(self.data.ndim).pattern}",
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
        return self.clone(data=new_data, **self._select_variable_metadata(indices))

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

        idx_list = _index_list(indices, self.data.shape[self.layout.time])
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


@dataclass(frozen=True, kw_only=True)
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

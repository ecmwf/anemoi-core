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

import einops
import torch
from torch.distributed import ProcessGroup

from anemoi.models.data.flat import FlatSource
from anemoi.models.data.sources import FLATTEN_PATTERN
from anemoi.models.data.sources.base import _Source
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import check_shard_sizes_match_group
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed

LOGGER = logging.getLogger(__name__)


class GriddedSource(_Source):
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

        if isinstance(self.data, list):
            msg = f"{self.__class__.__name__} data must be a single tensor, not a list."
            raise TypeError(msg)

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

        current_pattern = self.layout.normalized(self.data.ndim).pattern
        flattened_data = einops.rearrange(self.data, f"{current_pattern} -> {FLATTEN_PATTERN}")

        flattened_coords = einops.repeat(
            self.coordinates,
            "grid latlon -> (batch ensemble grid) latlon",
            batch=self.batch_size,
            ensemble=self.ensemble_size,
        )
        # already on device; see Batch.to()

        return FlatSource(
            data=flattened_data,
            coordinates=flattened_coords,
            shard_sizes=self.shard_sizes,
            device=self.data.device,
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

    def apply_pairwise(
        self, other: "GriddedSource", func: Callable, *, per_sample_kwargs=None, **kwargs
    ) -> torch.Tensor:
        """Combine two gridded sources through ``func``."""
        if per_sample_kwargs is not None:
            raise ValueError("Gridded losses take batched arguments; per_sample_kwargs is only for tabular sources.")
        if not isinstance(other, GriddedSource):
            msg = f"Other source must be a GriddedSource; got {type(other).__name__}."
            raise TypeError(msg)
        if self.layout != other.layout:
            msg = f"Both sources must have the same layout; got {self.layout!r} and {other.layout!r}."
            raise ValueError(msg)
        # assert self.variables == other.variables, f"Both views must have the same variables; got {self.variables} and {other.variables}."
        if self.coordinates is None or other.coordinates is None:
            assert self.coordinates is other.coordinates, "Both views must agree on whether coordinates are available."
        else:
            assert torch.equal(self.coordinates, other.coordinates), "Both views must have the same coordinates."
        return func(
            self.data,
            other.data,
            layout=self.layout,
            statistics=self.statistics,
            name_to_index=self.name_to_index,
            **kwargs,
        )

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

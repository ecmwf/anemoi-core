# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations
import logging

from abc import ABC
from dataclasses import dataclass, replace
from typing import Any
from abc import abstractmethod
import torch

from anemoi.models.data.spec import SourceSpec
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.data.layout import TensorLayout


LOGGER = logging.getLogger(__name__)


def resolve_device(device: torch.device | str) -> torch.device:
    """Resolve ``device`` to a concrete device, filling in the current CUDA index.

    ``torch.device("cuda") != torch.device("cuda:0")``, so cache lookups keyed on a
    device need the index pinned down first.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _to_device(value, device, *, non_blocking: bool):
    """Recursively move tensors to ``device``, pass non-tensors through."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [_to_device(v, device, non_blocking=non_blocking) for v in value]
    return value


def _pin(value):
    """Recursively pin tensors, pass non-tensors through. See :func:`_to_device`."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, list):
        return [_pin(v) for v in value]
    return value


def _cached_static_coords(name, value, device, *, cache: dict, non_blocking: bool):
    """Return the device copy of a static coordinate tensor, transferring on first use.

    Static coordinates are constant for the whole run - the grid of a dataset is fixed
    by the graph nodes it is bound to - so a single H2D copy per dataset serves every
    batch. ``cache`` is owned by the caller (one per process) and populated here.

    Shape, dtype and device are still checked against ``value``, so a cache entry that
    does not describe this batch is refreshed rather than silently returned.
    """
    cached = cache.get(name)
    if (
        isinstance(cached, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and cached.device == device
        and cached.shape == value.shape
        and cached.dtype == value.dtype
    ):
        return cached

    if cached is not None:
        LOGGER.debug(
            "Static coordinates for %r no longer match the cached copy (cached %s on %s, got %s on %s); refreshing.",
            name,
            tuple(cached.shape),
            cached.device,
            tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__,
            value.device if isinstance(value, torch.Tensor) else "n/a",
        )

    moved = _to_device(value, device, non_blocking=non_blocking)
    if isinstance(moved, torch.Tensor):
        cache[name] = moved
    return moved


@dataclass(frozen=True, slots=True)
class _Source(ABC):
    """Per-dataset view returned by :meth:`Batch.view`.

    Bundles the per-dataset payload (data, coordinates, timedeltas) with
    its :class:`TensorLayout` so callers can index logical axes (``time``,
    ``variables``) without hard-coded dimension positions. The same API
    works for gridded and sparse observation datasets thanks to the
    ``layout.time_in_grid`` dispatch.
    """

    spec: SourceSpec
    data: torch.Tensor | list[torch.Tensor]
    coordinates: torch.Tensor | list[torch.Tensor] | None = None
    timedeltas: torch.Tensor | list[torch.Tensor] | None = None
    boundaries: list[tuple[slice, ...]] | None = None
    shard_sizes: ShardSizes | list[ShardSizes] = None

    def __post_init__(self) -> None:
        """Validate the payload against the spec that describes it.

        Metadata-only checks (unique variable names) live on
        :class:`~anemoi.models.data.spec.SourceSpec`; what remains here is
        everything that needs a materialized tensor.
        """
        if self.coordinates is None:
            msg = f"{self.__class__.__name__} {self.name!r} requires coordinates."
            raise ValueError(msg)

        samples = self.data if isinstance(self.data, list) else [self.data]

        for sample in samples:
            layout = self.layout.normalized(sample.ndim)
            n_channels = sample.shape[layout.variables]
            # A zero-width variables axis is a *template*: a payload that carries the
            # spec's shape but none of its channels, used to describe a source that is
            # still to be produced (see AnemoiTransportModelEncProcDec target
            # templates). Names then describe what the template is for, so they are
            # not required to match the absent channels.
            if n_channels != 0 and n_channels != len(self.variables):
                raise ValueError(
                    f"{self.__class__.__name__} {self.name!r} has {n_channels} variable channels "
                    f"but {len(self.variables)} names."
                )

        if samples and any(sample.dtype != samples[0].dtype for sample in samples):
            raise ValueError(f"{self.__class__.__name__} {self.name!r} requires the same dtype for every sample.")

    @property
    def name(self) -> str:
        """Dataset name."""
        return self.spec.name

    @property
    def variables(self) -> list[str]:
        """Variable names along the variables axis, in order."""
        return self.spec.variables

    @property
    def layout(self) -> TensorLayout:
        """Mapping from logical axes to physical dimension positions."""
        return self.spec.layout

    @property
    def statistics(self) -> dict[str, Any]:
        """Per-statistic arrays over the variable axis."""
        return self.spec.statistics

    @property
    def coordinates_are_static(self) -> bool:
        """Whether the coordinate tensor is fixed for the whole run."""
        return self.spec.coordinates_are_static

    @property
    def grid_size(self) -> int | None:
        """Full grid size before sharding; ``None`` for observation datasets."""
        return self.spec.grid_size

    @property
    def name_to_index(self) -> dict[str, int]:
        """Mapping from variable name to index along the variables axis.

        Memoised on the spec, so it survives repeated ``batch[name]`` access.
        """
        return self.spec.name_to_index

    def clone(self, **kwargs) -> "_Source":
        """Return a new view with replacements, sharing fields that are not replaced.

        To change what the spec says, replace the spec::

            source.clone(spec=source.spec.clone(variables=[...]))
        """
        return replace(self, **kwargs)

    def select(self, **kwargs) -> "_Source":
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
                raise ValueError(
                    f"Unsupported dimension for selection: {dim!r}. Supported dimensions are 'time' and 'variables'."
                )
        return source

    def axis_size(self, axis: str) -> int:
        """Return the logical size of ``axis`` for this source.

        Resolves what the layout does not spell out: for a tabular source the batch
        axis is the outer list, and an axis the layout does not materialise (e.g.
        ``ensemble`` on a source that has none) is an implicit singleton.
        """
        samples = self.data if isinstance(self.data, list) else [self.data]

        if axis == "batch" and isinstance(self.data, list):
            return len(self.data)

        position = getattr(self.layout, axis, None)
        if position is None:
            return 1
        if not samples:
            return 0
        return samples[0].shape[position]

    def contiguous(self) -> "_Source":
        """Return a new view whose underlying data tensors are contiguous."""
        return self.apply_func(lambda t, **_: t.contiguous())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
        static_coord_cache: dict[str, torch.Tensor] | None = None,
    ) -> "_Source":
        """Return a copy of this source with every tensor on ``device``.

        Data, coordinates and timedeltas move together; consumers rely on that, since
        :meth:`allgather` gathers coordinates alongside data in one collective and
        does not move them itself.

        When this source's coordinates are static and ``static_coord_cache`` is given,
        the coordinate tensor crosses to the device once per run rather than once per
        batch. The cache is owned by the caller, keyed by dataset name.
        """
        device = resolve_device(device)

        coordinates = self.coordinates
        if coordinates is not None:
            if self.coordinates_are_static and static_coord_cache is not None:
                coordinates = _cached_static_coords(
                    self.name, coordinates, device, cache=static_coord_cache, non_blocking=non_blocking
                )
            else:
                coordinates = _to_device(coordinates, device, non_blocking=non_blocking)

        return self.clone(
            data=_to_device(self.data, device, non_blocking=non_blocking),
            coordinates=coordinates,
            timedeltas=(
                None if self.timedeltas is None else _to_device(self.timedeltas, device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "_Source":
        """Return a copy with host memory pinned. Static coordinates are left untouched.

        Pinning static coordinates would buy nothing: with a ``static_coord_cache``
        (see :meth:`to`) they cross to the device once per run, not once per batch.
        """
        coordinates = self.coordinates
        if coordinates is not None and not self.coordinates_are_static:
            coordinates = _pin(coordinates)

        return self.clone(
            data=_pin(self.data),
            coordinates=coordinates,
            timedeltas=None if self.timedeltas is None else _pin(self.timedeltas),
        )

    @abstractmethod
    def flatten(self) -> "FlatSource":
        """Return a flat source (nodes, features)."""
        pass

    @abstractmethod
    def select_time(self, indices: slice | Sequence[int] | int) -> "_Source":
        """Return a new view restricted to the given time indices."""
        pass

    @abstractmethod
    def select_variables(self, indices: Sequence[int] | torch.Tensor | slice) -> "_Source":
        """Return a new view restricted to the given variable indices."""
        pass

    @abstractmethod
    def apply_func(self, func: Callable, in_place: bool = False, **kwargs) -> "_Source":
        """Apply a function to this view, returning a new view with the same metadata."""
        pass

    @abstractmethod
    def apply_pairwise(self, other: "_Source", func: Callable, **kwargs) -> torch.Tensor:
        """Combine this source with another through ``func``, returning a tensor.

        The pairwise counterpart of :meth:`apply_func`; a loss is the motivating
        case. Both sources must describe the same thing: same layout, same
        coordinates, same number of samples.
        """
        pass

    @abstractmethod
    def shard(self, group: ProcessGroup | None) -> "_Source":
        """Split this source across ``group`` along its grid axis.

        The inverse of :meth:`allgather`, and deliberately its neighbour: shard
        *descriptors* travel with the data, so the operations that create and
        consume them belong together rather than one being a static method on the
        model.

        Returns self when the source is already sharded, or when the group spans a
        single rank.
        """
        pass

    @abstractmethod
    def allgather(self, group: ProcessGroup | None) -> "_Source":
        """Allgather this view across the given process group.

        This is a collective operation that synchronizes all processes in
        the group. The view's data and coordinates are allgathered, while
        metadata like layout and variables are unchanged.

        shard_sizes is None means the view is replicated (not sharded), and
        implementations must then return self unchanged. allgather is therefore
        idempotent and safe to call defensively - callers rely on that, since the batch is
        gathered once in on_after_batch_transfer (when model.keep_batch_sharded is
        false) and again by the validation diagnostics.

        shard_sizes records one entry per rank of the group the view was sharded over,
        so gathering must use that same group; implementations validate this and raise
        ValueError on a mismatch rather than mis-gathering or hanging.

        Parameters
        ----------
        group : ProcessGroup or None
            The process group to allgather across. None means single-rank, i.e. the
            view is already complete.

        Returns
        -------
        _Source
            A new view with allgathered data, or self when already full-grid.
        """
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
        idx = torch.as_tensor(
            list(indices) if not isinstance(indices, torch.Tensor) else indices, dtype=torch.long, device=tensor.device
        )
        return tensor.index_select(var_dim, idx)


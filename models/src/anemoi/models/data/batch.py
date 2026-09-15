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
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ProcessGroup
from torch.utils.data import default_collate

from anemoi.models.data.sample import SourceSample
from anemoi.models.data.spec import SourceSpec
from anemoi.models.data.views import SourceView
from anemoi.models.data.views import TensorLayout
from anemoi.models.data.views import create_source_view

LOGGER = logging.getLogger(__name__)

IndicesType = slice | Sequence[int] | int


def _broadcast_to_dict(value, keys: Iterable[str]) -> dict[str, Any]:
    """Broadcast a non-dict value to a dict with the same value for each key."""
    if isinstance(value, dict):
        return value
    return {key: value for key in keys}


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of per-dataset sources.

    A batch is one mapping from dataset name to
    :class:`~anemoi.models.data.views.SourceView`. Each source owns its own
    payload (data, coordinates, timedeltas, shard sizes, boundaries) together with
    the :class:`~anemoi.models.data.spec.SourceSpec` that describes it, so there is
    a single place per dataset where that information lives.

    Per-dataset payload shapes are as the sources define them: gridded datasets hold
    one stacked tensor of shape ``(batch, time, ensemble, grid, vars)``; sparse
    observation datasets hold a ``list[torch.Tensor]`` of length ``batch``, one entry
    per sample. Coordinates are ``(N, 2)`` tensors stacking ``(latitudes, longitudes)``
    in **radians**, shared by reference for static grids.

    Every transformation returns a new batch; the receiver is never mutated.

    Read a dataset's payload through its source: ``batch["era5"].data``,
    ``batch["era5"].layout``, ``batch["era5"].variables``.
    """

    sources: dict[str, SourceView]

    # -- batch-level properties --------------------------------------------

    @property
    def spec(self) -> dict[str, SourceSpec]:
        """Per-dataset specs for this batch, without any of its data."""
        return {name: source.spec for name, source in self.sources.items()}

    def spec_for(self, dataset_name: str) -> SourceSpec:
        """Return the :class:`SourceSpec` describing one dataset in this batch."""
        return self[dataset_name].spec

    @property
    def size(self) -> int:
        """Number of samples (batch size) in this batch."""
        batch_sizes = {}
        for name, source in self.sources.items():
            if isinstance(source.data, list):
                batch_sizes[name] = len(source.data)
            else:
                batch_sizes[name] = source.data.shape[source.layout.batch]

        assert len(set(batch_sizes.values())) == 1, f"Inconsistent batch sizes across datasets: {batch_sizes}"
        return next(iter(batch_sizes.values()))

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Names of the datasets present in this batch (insertion order)."""
        return tuple(self.sources.keys())

    @property
    def device(self) -> torch.device:
        """Device the batch data lives on.

        Derived from the first dataset. All data tensors are expected to share a
        device after the call to :meth:`to`.
        """
        if not self.sources:
            raise ValueError("Cannot determine device of an empty batch.")
        return next(iter(self.sources.values())).device

    @property
    def static_coord_datasets(self) -> frozenset[str]:
        """Dataset names whose coordinate tensors are static."""
        return frozenset(name for name, source in self.sources.items() if source.coordinates_are_static)

    def is_static_coords(self, dataset_name: str) -> bool:
        """Return whether ``dataset_name``'s coordinates are static."""
        return dataset_name in self.sources and self.sources[dataset_name].coordinates_are_static

    def __repr__(self) -> str:
        """Compact summary of per-dataset shapes, layouts and static-coords flag."""
        if not self.sources:
            return "Batch(<empty>)"

        lines = ["Batch("]
        for name, source in self.sources.items():
            if isinstance(source.data, list):
                shapes = [tuple(t.shape) for t in source.data]
                shape_repr = f"list[{len(source.data)}] of shapes={shapes}"
            else:
                shape_repr = f"shape={tuple(source.data.shape)}"
            static_repr = " static_coords" if source.coordinates_are_static else ""
            shard_repr = f" shard_sizes={source.shard_sizes}" if source.shard_sizes is not None else ""
            lines.append(f"  {name}: {shape_repr} layout={source.layout!r}{static_repr}{shard_repr}")
        lines.append(")")
        return "\n".join(lines)

    # -- mapping protocol ---------------------------------------------------
    # Implemented structurally rather than by inheriting collections.abc.Mapping,
    # whose ``__eq__`` mixin would compare batches element-wise and so raise on
    # tensor payloads.

    def __getitem__(self, dataset_name: str) -> SourceView:
        """Return the source for one dataset."""
        try:
            return self.sources[dataset_name]
        except KeyError:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.sources)})."
            raise KeyError(msg) from None

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self.sources

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> Iterator[str]:
        return iter(self.sources)

    def get(self, dataset_name: str, default: Any = None) -> SourceView | Any:
        """Return the source for ``dataset_name``, or ``default`` if absent."""
        return self.sources.get(dataset_name, default)

    def keys(self):  # noqa: D401 - mapping protocol
        """Return the dataset names (mapping protocol)."""
        return self.sources.keys()

    def values(self):  # noqa: D401 - mapping protocol
        """Return the per-dataset sources (mapping protocol)."""
        return self.sources.values()

    def items(self):  # noqa: D401 - mapping protocol
        """Return ``(name, source)`` pairs (mapping protocol)."""
        return self.sources.items()

    # -- transformations ----------------------------------------------------

    def with_sources(self, sources: dict[str, SourceView]) -> "Batch":
        """Return a new batch wrapping ``sources``."""
        return Batch(sources=sources)

    def replace(self, source_name: str, source: SourceView) -> "Batch":
        """Return a new batch with one dataset replaced."""
        return Batch(sources={**self.sources, source_name: source})

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
        static_coord_cache: dict[str, torch.Tensor] | None = None,
    ) -> "Batch":
        """Move the batch to ``device``.

        Every tensor in the returned batch - data, coordinates and timedeltas - lives
        on ``device``. Consumers rely on that: :meth:`SourceView.allgather` gathers
        coordinates alongside data in one collective and does not move them itself.

        Passing ``static_coord_cache`` transfers each static coordinate tensor once
        per run and reuses that device copy for every later batch. Without a cache
        they are transferred like anything else - correct, one small H2D copy per
        batch.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, optional
            Passed to :meth:`torch.Tensor.to`, by default True.
        static_coord_cache : dict[str, torch.Tensor], optional
            Caller-owned cache of static coordinates already on ``device``, keyed by
            dataset name. Populated on first use and mutated in place.

        Returns
        -------
        Batch
            A new batch on ``device``; the receiver is not mutated.
        """
        return Batch(
            sources={
                name: source.to(device, non_blocking=non_blocking, static_coord_cache=static_coord_cache)
                for name, source in self.sources.items()
            }
        )

    def pin_memory(self) -> "Batch":
        """Pin host memory for non-static tensors. Static coords are left untouched."""
        return Batch(sources={name: source.pin_memory() for name, source in self.sources.items()})

    def with_data(self, new_data: dict[str, torch.Tensor | list[torch.Tensor]]) -> "Batch":
        """Return a new :class:`Batch` with the data payloads replaced.

        Everything else about each source - coordinates, timedeltas, spec - is shared
        by reference, which preserves static-coord identity (no extra H2D, no copy).
        Passing a subset of the dataset names narrows the batch to those datasets.

        Parameters
        ----------
        new_data : dict[str, torch.Tensor | list[torch.Tensor]]
            Replacement data payloads, keyed by dataset name.

        Returns
        -------
        Batch
            A new frozen :class:`Batch` sharing this batch's envelope by reference.
        """
        unknown_keys = set(new_data) - set(self.sources)
        if unknown_keys:
            msg = f"Replacement data contains unknown dataset names: {sorted(unknown_keys)}."
            raise ValueError(msg)

        return Batch(sources={name: self.sources[name].clone(data=payload) for name, payload in new_data.items()})

    def apply(self, func: Callable, **kwargs) -> "Batch":
        """Return a new batch with ``func`` applied to every source's data."""
        return Batch(sources={name: source.apply_func(func, **kwargs) for name, source in self.sources.items()})

    def apply_pairwise(self, other: "Batch", func: Callable, **kwargs) -> dict[str, torch.Tensor]:
        """Apply ``func`` to each ``(self[name], other[name])`` pair.

        The per-dataset counterpart of a loss over two batches::

            batch.apply_pairwise(target, loss_fn)

        Returns
        -------
        dict[str, torch.Tensor]
            One result per dataset name, as returned by
            :meth:`SourceView.apply_loss`.
        """
        missing = set(self.sources) - set(other.sources)
        if missing:
            msg = f"Other batch is missing dataset(s) {sorted(missing)}."
            raise ValueError(msg)
        return {
            name: source.apply_loss(other[name], func, **kwargs) for name, source in self.sources.items()
        }

    def select(self, **kwargs) -> "Batch":
        """Return a new :class:`Batch` with per-dataset selection applied.

        Each value may be a plain index applied to every dataset, or a
        ``dict[dataset_name, index]``.
        """
        per_source_indices = defaultdict(dict)
        for dim, indices in kwargs.items():
            # if indices is not a dict, broadcast the same indexing to every dataset.
            indices_dict = _broadcast_to_dict(indices, self.dataset_names)
            for source_name, idx in indices_dict.items():
                per_source_indices[source_name][dim] = idx

        new_sources = dict(self.sources)
        for source_name, per_source_idx in per_source_indices.items():
            new_sources[source_name] = self.sources[source_name].select(**per_source_idx)

        return Batch(sources=new_sources)

    def allgather(self, group: ProcessGroup | None) -> "Batch":
        """Allgather the batch across the given process group.

        This is a collective operation that synchronizes all processes in ``group``.
        All processes must call it with the same group and have batches of the same
        size and dataset structure.

        Idempotent: datasets that are already full-grid (``shard_sizes is None``) are
        left untouched.

        Parameters
        ----------
        group : ProcessGroup or None
            The process group to allgather across.

        Returns
        -------
        Batch
            A new Batch with allgathered data, or self if nothing was sharded.
        """
        new_sources = {name: source.allgather(group=group) for name, source in self.sources.items()}
        if all(new is old for new, old in zip(new_sources.values(), self.sources.values())):
            return self
        return Batch(sources=new_sources)

    @staticmethod
    def collate(samples: list[dict[str, SourceSample]] | dict[str, SourceSample]) -> "Batch":
        """Collate per-sample :class:`SourceSample` payloads into a :class:`Batch`.

        Each sample is a mapping ``{dataset_name: SourceSample}``. The sample itself
        says how it must be collated, so there is no side channel:

        * **Gridded** (``layout.time_in_grid`` false) - every sample has the same
          shape, so data and (non-static) coordinates are stacked along a new leading
          batch axis via :func:`torch.utils.data.default_collate`, and the layout is
          shifted with :meth:`TensorLayout.with_batch_dim`. A sample whose
          ``coordinates_are_static`` is set reuses the first sample's coordinate
          tensor by reference - no stacking, no copy.
        * **Tabular** (``layout.time_in_grid`` true) - the grid extent varies per
          sample, so data, coordinates, timedeltas, boundaries and shard sizes each
          become a list of length ``B`` and the per-sample layout stands, the batch
          axis being the list itself.
        """
        if isinstance(samples, dict):
            samples = [samples]

        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        # Discover the dataset names from the first sample; assume consistent.
        first = samples[0]

        sources: dict[str, SourceView] = {}
        for name, head in first.items():
            per_sample = [sample[name] for sample in samples]

            if head.is_tabular:
                data: Any = [s.data for s in per_sample]
                coordinates = None if head.coordinates is None else [s.coordinates for s in per_sample]
                timedeltas = None if head.timedeltas is None else [s.timedeltas for s in per_sample]
                boundaries = [s.boundaries for s in per_sample]
                shard_sizes = None if head.shard_sizes is None else [s.shard_sizes for s in per_sample]
                layout = head.layout
            else:
                data = default_collate([s.data for s in per_sample])
                if head.coordinates is None:
                    coordinates = None
                elif head.coordinates_are_static:
                    coordinates = head.coordinates
                else:
                    coordinates = default_collate([s.coordinates for s in per_sample])
                timedeltas = (
                    None if head.timedeltas is None else default_collate([s.timedeltas for s in per_sample])
                )
                boundaries = None
                shard_sizes = head.shard_sizes
                layout = head.layout.with_batch_dim()

            _validate_layout_against(name, layout, data)

            sources[name] = create_source_view(
                spec=SourceSpec(
                    name=name,
                    variables=head.variables,
                    layout=layout,
                    statistics=head.statistics,
                    grid_size=head.grid_size,
                    coordinates_are_static=head.coordinates_are_static,
                ),
                data=data,
                coordinates=coordinates,
                timedeltas=timedeltas,
                boundaries=boundaries,
                shard_sizes=shard_sizes,
            )

        batch = Batch(sources)
        LOGGER.debug("Batch.collate produced:\n%r", batch)
        return batch


def _validate_layout_against(name: str, layout: TensorLayout, data: torch.Tensor | list[torch.Tensor]) -> None:
    """Check every non-None axis position is a valid axis of the collated tensor.

    Catches reader-side mistakes early instead of letting them surface as cryptic
    errors deep inside model code.
    """
    ref = data[0] if isinstance(data, list) else data
    ndim = ref.ndim
    for axis_name in TensorLayout._AXIS:
        pos = getattr(layout, axis_name)
        if pos is None:
            continue
        if not (-ndim <= pos < ndim):
            msg = (
                f"TensorLayout for dataset {name!r} declares {axis_name}={pos} but the "
                f"collated tensor only has {ndim} dimensions (shape={tuple(ref.shape)}). "
                f"Layout: {layout!r}."
            )
            raise ValueError(msg)

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
from typing import Self

import torch
from rich.console import Console
from rich.tree import Tree
from torch.distributed import ProcessGroup

from anemoi.models.data.sample import BaseSample
from anemoi.models.data.sources.base import BaseTemplate
from anemoi.models.data.sources.base import Source

LOGGER = logging.getLogger(__name__)

IndicesType = slice | Sequence[int] | int


def _broadcast_to_dict(value, keys: Iterable[str]) -> dict[str, Any]:
    """Broadcast a non-dict value to a dict with the same value for each key."""
    if isinstance(value, dict):
        return value
    return {key: value for key in keys}


@dataclass(frozen=True, eq=False, slots=True)
class Batch:
    """A batch of per-dataset sources.

    A batch is one mapping from dataset name to
    :class:`~anemoi.models.data.source.Source`. Each source owns its own
    payload (data, coordinates, timedeltas, shard sizes, boundaries) together with
    the metadata that describes it (name, variables, layout, statistics), so there
    is a single place per dataset where that information lives.

    Per-dataset payload shapes are as the sources define them: gridded datasets hold
    one stacked tensor of shape ``(batch, time, ensemble, grid, vars)``; sparse
    observation datasets hold a ``list[torch.Tensor]`` of length ``batch``, one entry
    per sample. Coordinates are ``(N, 2)`` tensors stacking ``(latitudes, longitudes)``
    in **radians**, shared by reference for static grids.

    Every transformation returns a new batch; the receiver is never mutated.

    Read a dataset's payload through its source: ``batch["era5"].data``,
    ``batch["era5"].layout``, ``batch["era5"].variables``.
    """

    sources: dict[str, Source]

    def template(self) -> dict[str, BaseTemplate]:
        """Return every source without its data, keyed by dataset name (see :meth:`Source.template`)."""
        return {name: source.template() for name, source in self.sources.items()}

    @property
    def batch_size(self) -> int:
        batch_sizes = {name: source.batch_size for name, source in self.sources.items()}
        if not batch_sizes:
            msg = "Cannot determine batch size of an empty batch."
            raise ValueError(msg)

        if len(set(batch_sizes.values())) != 1:
            msg = f"Inconsistent batch sizes across datasets: {batch_sizes}"
            raise ValueError(msg)

        return next(iter(batch_sizes.values()))

    @property
    def ensemble_size(self) -> int:
        ensemble_sizes = {name: source.ensemble_size for name, source in self.sources.items()}
        if not ensemble_sizes:
            raise ValueError("Cannot determine ensemble size of an empty batch.")

        if len(set(ensemble_sizes.values())) != 1:
            raise ValueError(f"Inconsistent ensemble sizes across datasets: {ensemble_sizes}")

        return next(iter(ensemble_sizes.values()))

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

    def tree(self) -> Tree:
        """Return a tree representation of the batch."""
        if not self.sources:
            return Tree("Batch(<empty>)")

        tree = Tree("Batch")
        for source in self.sources.values():
            tree.add(source.tree())

        return tree

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def __getitem__(self, dataset_name: str) -> Source:
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

    def get(self, dataset_name: str, default: Any = None) -> Source | Any:
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

    def with_sources(self, sources: dict[str, Source]) -> "Batch":
        """Return a new batch wrapping ``sources``."""
        return Batch(sources=sources)

    def replace(self, source_name: str, source: Source) -> "Batch":
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
        on ``device``. Consumers rely on that: :meth:`Source.allgather` gathers
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

        Everything else about each source - coordinates, timedeltas, variables, statistics - is shared
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

    def map_data(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        """Return a new batch with ``fn`` applied to every dataset's payload (see :meth:`Source.map_data`)."""
        return Batch(sources={name: source.map_data(fn) for name, source in self.sources.items()})

    def zip_map_data(self, fn: Callable[..., torch.Tensor], *others: "Batch") -> "Batch":
        """Return a new batch with ``fn`` applied dataset by dataset to this batch and ``others``.

        Every batch in ``others`` must contain this batch's datasets (see :meth:`Source.zip_map_data`).
        """
        return Batch(
            sources={
                name: source.zip_map_data(fn, *(other[name] for other in others))
                for name, source in self.sources.items()
            },
        )

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
    def collate(samples: list[dict[str, BaseSample]] | dict[str, BaseSample]) -> Self:
        """Collate per-sample :class:`BaseSample` payloads into a :class:`Batch`.

        Each sample is a mapping ``{dataset_name: BaseSample}``. The class of each
        dataset's sample decides how it is collated (see
        :meth:`GriddedSample.collate` and :meth:`TabularSample.collate`),
        so every sample of a dataset must be of the same class.
        """
        if isinstance(samples, dict):
            samples = [samples]

        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        # Discover the dataset names from the first sample
        first = samples[0]

        sources: dict[str, Source] = {}
        for name, head in first.items():
            per_sample = [sample[name] for sample in samples]
            sample_cls = type(head)
            if not isinstance(head, BaseSample) or any(type(s) is not sample_cls for s in per_sample):
                kinds = sorted({type(s).__name__ for s in per_sample})
                msg = f"Dataset {name!r} must be collated from a single BaseSample subclass; got {kinds}."
                raise TypeError(msg)

            sources[name] = sample_cls.collate(name, per_sample)

        batch = Batch(sources)
        LOGGER.debug("Batch.collate produced:\n%r", batch)
        return batch

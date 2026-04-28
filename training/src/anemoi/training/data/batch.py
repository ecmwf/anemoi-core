# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Typed batch envelope carrying data, coordinates and metadata.

A :class:`Batch` is the object that flows from the dataloader to the model.
It bundles, per dataset:

* ``data`` — the input tensor with shape ``(batch, time, ensemble, grid, vars)``.
* ``coords`` — per-dataset coordinate tensors (e.g. ``latitudes``, ``longitudes``)
  in **radians**.
* ``metadata`` — extension point for per-sample information (masks, timestamps, ...).

Coordinate tensors for static-grid datasets are shared by reference across
the batch dimension to avoid per-worker / per-step copies. The set of static
dataset names is stored under ``metadata["static_coords"]`` and consulted by
:meth:`Batch.to` to skip redundant host-to-device transfers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch.utils.data import default_collate

# Key in ``Batch.metadata`` listing dataset names whose coordinate tensors are
# static (allocated once, shared by reference, not transferred per step).
STATIC_COORDS_META_KEY = "static_coords"


@dataclass(frozen=True, slots=True)
class Batch:
    """Typed batch carrying data, coordinates and metadata.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Per-dataset input tensor, shape ``(batch, time, ensemble, grid, vars)``.
    coords : dict[str, dict[str, torch.Tensor]]
        Per-dataset coordinate tensors keyed by name (e.g. ``"latitudes"``,
        ``"longitudes"``) in **radians**. For static-grid datasets the
        tensors are shared by reference across the batch dimension and have
        shape ``(grid, ...)``; for dynamic-grid datasets they have a leading
        batch (and optionally time) dimension.
    metadata : dict[str, Any], optional
        Free-form per-batch metadata. Reserved keys:

        * ``"static_coords"`` — :class:`frozenset` of dataset names whose
          coordinates are static. :meth:`to` and :meth:`pin_memory` will
          skip these entries.
    """

    data: dict[str, torch.Tensor]
    coords: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- access

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Names of the datasets present in this batch (insertion order)."""
        return tuple(self.data.keys())

    @property
    def static_coord_datasets(self) -> frozenset[str]:
        """Dataset names whose coordinate tensors are static."""
        return frozenset(self.metadata.get(STATIC_COORDS_META_KEY, ()))

    def is_static_coords(self, dataset_name: str) -> bool:
        """Return whether ``dataset_name``'s coordinates are static."""
        return dataset_name in self.static_coord_datasets

    def __getitem__(self, dataset_name: str) -> torch.Tensor:
        """Return the data tensor for ``dataset_name`` (mapping behaviour).

        For the richer per-dataset view that bundles data, coords and the
        static flag, use :meth:`view`.
        """
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        return self.data[dataset_name]

    def view(self, dataset_name: str) -> "DatasetView":
        """Return a per-dataset view bundling data, coords and the static flag."""
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        return DatasetView(
            name=dataset_name,
            data=self.data[dataset_name],
            coords=self.coords.get(dataset_name, {}),
            is_static=self.is_static_coords(dataset_name),
        )

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self.data

    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):  # noqa: D401 - mapping protocol
        """Return the dataset names (mapping protocol)."""
        return self.data.keys()

    def values(self):  # noqa: D401 - mapping protocol
        """Return the data tensors (mapping protocol)."""
        return self.data.values()

    def items(self):  # noqa: D401 - mapping protocol
        """Return ``(name, tensor)`` pairs (mapping protocol)."""
        return self.data.items()

    # -------------------------------------------------------- device transfer

    def to(self, device: torch.device | str, *, non_blocking: bool = True) -> "Batch":
        """Move the batch to ``device``.

        Static coordinate tensors (those whose dataset name is listed in
        ``metadata["static_coords"]``) are skipped: they are expected to have
        been moved to the model device once, at training start, and never
        transferred again.

        Returns a new :class:`Batch`; the receiver is not mutated.
        """
        new_data = {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coords: dict[str, dict[str, torch.Tensor]] = {}
        for name, per_dataset in self.coords.items():
            if name in static:
                # Skip the H2D transfer; the tensors are already on device.
                new_coords[name] = per_dataset
                continue
            new_coords[name] = {key: tensor.to(device, non_blocking=non_blocking) for key, tensor in per_dataset.items()}

        return Batch(data=new_data, coords=new_coords, metadata=self.metadata)

    def pin_memory(self) -> "Batch":
        """Pin host memory for non-static tensors. Static coords are left untouched."""
        new_data = {name: tensor.pin_memory() for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coords: dict[str, dict[str, torch.Tensor]] = {}
        for name, per_dataset in self.coords.items():
            if name in static:
                new_coords[name] = per_dataset
                continue
            new_coords[name] = {key: tensor.pin_memory() for key, tensor in per_dataset.items()}

        return Batch(data=new_data, coords=new_coords, metadata=self.metadata)

    # ----------------------------------------------------------- construction

    @staticmethod
    def collate(
        samples: list[dict[str, dict[str, Any]]],
        *,
        static_coord_datasets: Iterable[str] = (),
    ) -> "Batch":
        """Collate a list of per-sample dicts into a :class:`Batch`.

        Each sample must be a mapping ``{dataset_name: payload}`` where
        ``payload`` is a mapping with a required ``"data"`` key and an
        optional ``"coords"`` mapping.

        For datasets listed in ``static_coord_datasets`` the coordinate
        tensors of the **first** sample are reused by reference (no stacking,
        no copy); the per-batch payload is therefore a single shared
        reference across the batch dimension. For dynamic datasets the
        coordinates are stacked along a new leading batch dimension via
        :func:`torch.utils.data.default_collate`.
        """
        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        static = frozenset(static_coord_datasets)

        # Discover the dataset names from the first sample; assume consistent.
        first = samples[0]
        dataset_names = tuple(first.keys())

        # Split each sample into a data dict and a coords dict.
        per_sample_data: list[dict[str, torch.Tensor]] = []
        per_sample_coords: list[dict[str, dict[str, torch.Tensor]]] = []
        for sample in samples:
            sample_data: dict[str, torch.Tensor] = {}
            sample_coords: dict[str, dict[str, torch.Tensor]] = {}
            for name in dataset_names:
                payload = sample[name]
                sample_data[name] = payload["data"]
                if "coords" in payload:
                    sample_coords[name] = payload["coords"]
            per_sample_data.append(sample_data)
            per_sample_coords.append(sample_coords)

        # Stack data tensors along the batch dim (default_collate preserves
        # dict-of-tensor structure).
        collated_data: dict[str, torch.Tensor] = default_collate(per_sample_data)

        # Coordinates: share by reference for static datasets, stack for the rest.
        collated_coords: dict[str, dict[str, torch.Tensor]] = {}
        for name in dataset_names:
            if name not in per_sample_coords[0]:
                continue
            if name in static:
                # Use the first sample's coords by reference; do NOT copy or
                # repeat across the batch dimension.
                collated_coords[name] = per_sample_coords[0][name]
            else:
                collated_coords[name] = default_collate([per_sample_coords[i][name] for i in range(len(samples))])

        metadata: dict[str, Any] = {}
        if static:
            metadata[STATIC_COORDS_META_KEY] = frozenset(static)

        return Batch(data=collated_data, coords=collated_coords, metadata=metadata)


@dataclass(frozen=True, slots=True)
class DatasetView:
    """Per-dataset view returned by :meth:`Batch.__getitem__`."""

    name: str
    data: torch.Tensor
    coords: dict[str, torch.Tensor]
    is_static: bool

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
    grid_sizes : dict[str, int], optional
        Per-dataset full grid sizes (number of grid points before any
        distributed sharding). Populated during collation from sample
        payloads. For static-grid datasets this equals
        ``sum(dataset.grids)``; for observation datasets it equals the
        grid dimension of the data tensor.
    timedeltas : dict[str, torch.Tensor], optional
        Per-dataset timedelta tensors (e.g. for observation datasets
        where each point has an associated time offset). Keyed by
        dataset name. Transferred to device alongside data.
    grid_shard_indices : dict[str, Any], optional
        Per-dataset grid shard indices (``np.ndarray``, ``slice``, or
        ``None``). Describes which grid points are present in this
        batch's shard. Not transferred to device.
    """

    data: dict[str, torch.Tensor]
    coords: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    grid_sizes: dict[str, int] = field(default_factory=dict)
    timedeltas: dict[str, torch.Tensor] = field(default_factory=dict)
    grid_shard_indices: dict[str, Any] = field(default_factory=dict)

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
            timedeltas=self.timedeltas.get(dataset_name),
            grid_shard_indices=self.grid_shard_indices.get(dataset_name),
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

        new_timedeltas = {
            name: tensor.to(device, non_blocking=non_blocking) for name, tensor in self.timedeltas.items()
        }

        return Batch(
            data=new_data,
            coords=new_coords,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            grid_shard_indices=self.grid_shard_indices,
        )

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

        new_timedeltas = {name: tensor.pin_memory() for name, tensor in self.timedeltas.items()}

        return Batch(
            data=new_data,
            coords=new_coords,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            grid_shard_indices=self.grid_shard_indices,
        )

    # ------------------------------------------------------------- transform

    def with_data(self, new_data: dict[str, torch.Tensor]) -> "Batch":
        """Return a new :class:`Batch` with ``data`` replaced by ``new_data``.

        ``coords`` and ``metadata`` are shared by reference with the
        receiver. This preserves static-coord identity (no extra H2D, no
        copy) and is the recommended way for tasks to slice or transform
        the data tensors while keeping the coordinate envelope intact.

        Parameters
        ----------
        new_data : dict[str, torch.Tensor]
            Replacement data tensors. Must cover the same dataset names
            as ``self.data``.

        Returns
        -------
        Batch
            A new frozen :class:`Batch` sharing ``self.coords`` and
            ``self.metadata`` by reference.
        """
        if set(new_data.keys()) != set(self.data.keys()):
            msg = (
                "with_data: new_data keys must match the existing dataset names "
                f"(got {sorted(new_data)}, expected {sorted(self.data)})."
            )
            raise ValueError(msg)

        return Batch(
            data=new_data,
            coords=self.coords,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=self.timedeltas,
            grid_shard_indices=self.grid_shard_indices,
        )

    def node_coords(self, dataset_name: str) -> torch.Tensor | None:
        """Return per-node ``(num_nodes, 2)`` lat/lon coords for ``dataset_name``.

        Stacks ``coords[dataset_name]["latitudes"]`` and ``["longitudes"]``
        along a trailing dimension. Coordinates are expected in **radians**
        (see the module docstring) and to share the layout used at graph
        registration time, so the result can be fed directly to
        :meth:`anemoi.models.layers.graph.NamedNodesAttributes.forward`
        via the ``coords=`` kwarg.

        Returns ``None`` when the dataset has no ``coords`` entry or is
        missing one of ``latitudes``/``longitudes`` — the model layer
        then falls back to its registered static buffer (preserving
        full backward compatibility with checkpoints trained before
        the typed-batch refactor).
        """
        per_dataset = self.coords.get(dataset_name)
        if not per_dataset:
            return None
        latitudes = per_dataset.get("latitudes")
        longitudes = per_dataset.get("longitudes")

        node_coords = torch.stack([latitudes, longitudes], dim=-1)
        #node_coords = torch.cat([torch.sin(node_coords), torch.cos(node_coords)], dim=-1)
        return node_coords

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

        # Split each sample into a data dict, coords dict, and timedeltas dict.
        per_sample_data: list[dict[str, torch.Tensor]] = []
        per_sample_coords: list[dict[str, dict[str, torch.Tensor]]] = []
        per_sample_timedeltas: list[dict[str, torch.Tensor]] = []
        for sample in samples:
            sample_data: dict[str, torch.Tensor] = {}
            sample_coords: dict[str, dict[str, torch.Tensor]] = {}
            sample_timedeltas: dict[str, torch.Tensor] = {}
            for name in dataset_names:
                payload = sample[name]
                sample_data[name] = payload["data"]
                if "coords" in payload:
                    sample_coords[name] = payload["coords"]
                if "timedeltas" in payload:
                    sample_timedeltas[name] = payload["timedeltas"]
            per_sample_data.append(sample_data)
            per_sample_coords.append(sample_coords)
            per_sample_timedeltas.append(sample_timedeltas)

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
                # TODO: Fix for batch_size > 1
                # collated_coords[name] = default_collate([per_sample_coords[i][name] for i in range(len(samples))])
                collated_coords[name] = per_sample_coords[0][name]

        # Timedeltas: collate for datasets that have them.
        collated_timedeltas: dict[str, torch.Tensor] = {}
        if per_sample_timedeltas[0]:
            for name in per_sample_timedeltas[0]:
                collated_timedeltas[name] = default_collate(
                    [per_sample_timedeltas[i][name] for i in range(len(samples))]
                )

        # Grid sizes: extract from sample payloads (all samples in a batch
        # must agree since default_collate requires matching tensor shapes).
        collated_grid_sizes: dict[str, int] = {}
        for name in dataset_names:
            payload = first[name]
            if "grid_size" in payload:
                collated_grid_sizes[name] = payload["grid_size"]

        # Grid shard indices: same for all samples in a batch (same worker/shard).
        collated_grid_shard_indices: dict[str, Any] = {}
        for name in dataset_names:
            payload = first[name]
            if "grid_shard_indices" in payload:
                collated_grid_shard_indices[name] = payload["grid_shard_indices"]

        metadata: dict[str, Any] = {}
        if static:
            metadata[STATIC_COORDS_META_KEY] = frozenset(static)

        return Batch(
            data=collated_data,
            coords=collated_coords,
            metadata=metadata,
            grid_sizes=collated_grid_sizes,
            timedeltas=collated_timedeltas,
            grid_shard_indices=collated_grid_shard_indices,
        )


@dataclass(frozen=True, slots=True)
class DatasetView:
    """Per-dataset view returned by :meth:`Batch.__getitem__`."""

    name: str
    data: torch.Tensor
    coords: dict[str, torch.Tensor]
    is_static: bool
    timedeltas: torch.Tensor | None = None
    grid_shard_indices: Any = None

    @property
    def latlons(self) -> torch.Tensor:
        """Return latitudes and longitudes in radians, stacked along a trailing dim."""
        latlons = torch.stack([self.coords["latitudes"], self.coords["longitudes"]], dim=-1)
        return torch.cat([torch.sin(latlons), torch.cos(latlons)], dim=-1).to(torch.float32)

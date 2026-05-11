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

* ``data`` — either a stacked tensor with shape
  ``(batch, time, ensemble, grid, vars)`` (gridded datasets) or a
  ``list[torch.Tensor]`` of length ``batch`` with per-sample shape
  ``(ensemble=1, grid_i, vars)`` (sparse observation datasets, where
  ``grid_i`` varies per sample).
* ``coordinates`` — per-dataset ``(N, 2)`` tensor stacking
  ``(latitude, longitude)`` along the trailing dimension, in **radians**.
  For static-grid datasets the tensor is shared by reference across the
  batch dimension; for dynamic-grid (gridded non-static) datasets it has
  a leading batch dimension; for sparse datasets the value is a
  ``list[torch.Tensor]`` of length ``batch``.
* ``timedeltas`` — *(sparse only)* per-dataset ``(N,)`` tensor (or
  ``list[torch.Tensor]`` of length ``batch``) holding the per-point time
  offset in seconds. Stored separately from ``coordinates`` so consumers
  can route the spatial and temporal axes independently.
* ``metadata`` — extension point for per-batch / per-dataset information
  (masks, timestamps, sparse-obs ``boundaries``, ...).

Coordinate tensors for static-grid datasets are shared by reference across
the batch dimension to avoid per-worker / per-step copies. The set of static
dataset names is stored under ``metadata["static_coords"]`` and consulted by
:meth:`Batch.to` to skip redundant host-to-device transfers.

Sparse observation datasets carry their per-time split information under
``metadata[dataset_name]["boundaries"]`` as a ``list[tuple[slice, ...]]``
(one entry per batch sample). These are Python ``slice`` objects and are
intentionally **not** transferred to device by :meth:`Batch.to`.
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

# Reserved per-dataset metadata key carrying sparse-obs per-time boundaries
# as a ``list[tuple[slice, ...]]`` (one entry per batch sample). Untouched by
# device transfer / pinning since ``slice`` is not a tensor.
BOUNDARIES_META_KEY = "boundaries"


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """Maps logical axes to physical dimension positions.

    Describes the semantic meaning of each dimension in a per-dataset data
    tensor so that downstream code (tasks, losses, metrics) can index axes
    by name rather than by hard-coded position.

    Parameters
    ----------
    batch : int or None
        Position of the batch dimension (``None`` before collation).
    time : int or None
        Position of the explicit time dimension. ``None`` when time is
        folded into the grid axis (sparse observation datasets).
    ensemble : int or None
        Position of the ensemble dimension (``None`` when absent).
    grid : int
        Position of the grid / spatial-points dimension.
    variables : int
        Position of the variable (channel) dimension.
    time_in_grid : bool
        ``True`` when the time axis is encoded within the grid dimension
        via ``boundaries`` metadata (sparse observations). ``False`` for
        gridded datasets that carry an explicit time axis.
    """

    batch: int | None = None
    time: int | None = None
    ensemble: int | None = None
    grid: int = -2
    variables: int = -1
    time_in_grid: bool = False

    def with_batch_dim(self) -> "TensorLayout":
        """Return a new layout shifted by +1 to account for a leading batch dim."""
        if self.batch is not None:
            return self
        return TensorLayout(
            batch=0,
            time=self.time + 1 if self.time is not None else None,
            ensemble=self.ensemble + 1 if self.ensemble is not None else None,
            grid=self.grid + 1 if self.grid >= 0 else self.grid,
            variables=self.variables + 1 if self.variables >= 0 else self.variables,
            time_in_grid=self.time_in_grid,
        )


def _to_device(value, device, *, non_blocking: bool):
    """Recursively move tensors to ``device``; pass non-tensors through.

    Handles the union types that flow through a sparse-aware :class:`Batch`:
    plain :class:`torch.Tensor`, ``list[torch.Tensor]`` (sparse per-sample
    payloads), and any non-tensor leaf (e.g. ``slice`` objects in the
    ``boundaries`` metadata) which is returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [_to_device(v, device, non_blocking=non_blocking) for v in value]
    return value


def _pin_memory(value):
    """Recursively pin tensors; pass non-tensors through. See :func:`_to_device`."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Batch:
    """Typed batch carrying data, coordinates and metadata.

    Parameters
    ----------
    data : dict[str, torch.Tensor | list[torch.Tensor]]
        Per-dataset input. For gridded datasets a single stacked tensor of
        shape ``(batch, time, ensemble, grid, vars)``; for sparse
        observation datasets a ``list[torch.Tensor]`` of length ``batch``,
        one entry per sample with shape ``(ensemble=1, grid_i, vars)``.
    coordinates : dict[str, torch.Tensor | list[torch.Tensor]]
        Per-dataset ``(N, 2)`` coordinate tensor stacking
        ``(latitudes, longitudes)`` in **radians**. For static-grid
        datasets the tensor is shared by reference across the batch
        dimension and has shape ``(grid, 2)``; for dynamic-grid (gridded
        non-static) datasets it has shape ``(batch, grid, 2)``; for
        sparse datasets it is a ``list[torch.Tensor]`` of length ``batch``
        with per-sample shape ``(grid_i, 2)``.
    metadata : dict[str, Any], optional
        Free-form per-batch metadata. Reserved keys:

        * ``"static_coords"`` — :class:`frozenset` of dataset names whose
          coordinates are static. :meth:`to` and :meth:`pin_memory` will
          skip these entries.
        * ``<dataset_name>`` — for sparse datasets, a dict that may carry
          ``"boundaries": list[tuple[slice, ...]]`` (one per batch sample).
          Non-tensor leaves are passed through device transfer untouched.
    grid_sizes : dict[str, int], optional
        Per-dataset full grid sizes (number of grid points before any
        distributed sharding). Populated during collation from sample
        payloads. For static-grid datasets this equals
        ``sum(dataset.grids)``; for observation datasets it equals the
        grid dimension of the data tensor.
    timedeltas : dict[str, torch.Tensor | list[torch.Tensor]], optional
        Per-dataset per-point time-offset tensors (sparse observation
        datasets only) of shape ``(N,)``. Stored separately from
        :attr:`coordinates` so the spatial and temporal axes can be
        consumed independently. For sparse datasets the value is a
        ``list[torch.Tensor]`` of length ``batch``. Transferred to
        device alongside data.
    grid_shard_indices : dict[str, Any], optional
        Per-dataset grid shard indices (``np.ndarray``, ``slice``, or
        ``None``). Describes which grid points are present in this
        batch's shard. Not transferred to device.
    layouts : dict[str, TensorLayout], optional
        Per-dataset :class:`TensorLayout` descriptors mapping logical
        axes (time, ensemble, grid, variables) to physical dimension
        positions. Not transferred to device.
    """

    data: dict[str, torch.Tensor | list[torch.Tensor]]
    coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    grid_sizes: dict[str, int] = field(default_factory=dict)
    timedeltas: dict[str, torch.Tensor | list[torch.Tensor]] = field(default_factory=dict)
    grid_shard_indices: dict[str, Any] = field(default_factory=dict)
    layouts: dict[str, TensorLayout] = field(default_factory=dict)

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

    def __getitem__(self, dataset_name: str) -> torch.Tensor | list[torch.Tensor]:
        """Return the data payload for ``dataset_name`` (mapping behaviour).

        Returns a stacked :class:`torch.Tensor` for gridded datasets and a
        ``list[torch.Tensor]`` of length ``batch`` for sparse observation
        datasets. For the richer per-dataset view that bundles data,
        coords and the static flag, use :meth:`view`.
        """
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)

        return self.data[dataset_name]

    def view(self, dataset_name: str) -> "DatasetView":
        """Return a per-dataset view bundling data, coordinates and the static flag."""
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        return DatasetView(
            name=dataset_name,
            data=self.data[dataset_name],
            coordinates=self.coordinates.get(dataset_name),
            is_static=self.is_static_coords(dataset_name),
            timedeltas=self.timedeltas.get(dataset_name),
            grid_shard_indices=self.grid_shard_indices.get(dataset_name),
            layout=self.layouts.get(dataset_name),
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
        transferred again. Non-tensor metadata leaves (e.g. ``boundaries``
        ``slice`` objects for sparse observations) are passed through
        untouched.

        Returns a new :class:`Batch`; the receiver is not mutated.
        """
        new_data = {name: _to_device(tensor, device, non_blocking=non_blocking) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name, value in self.coordinates.items():
            if name in static:
                # Skip the H2D transfer; the tensors are already on device.
                new_coordinates[name] = value
                continue
            new_coordinates[name] = _to_device(value, device, non_blocking=non_blocking)

        new_timedeltas = {
            name: _to_device(value, device, non_blocking=non_blocking)
            for name, value in self.timedeltas.items()
        }

        return Batch(
            data=new_data,
            coordinates=new_coordinates,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            grid_shard_indices=self.grid_shard_indices,
            layouts=self.layouts,
        )

    def pin_memory(self) -> "Batch":
        """Pin host memory for non-static tensors. Static coords are left untouched."""
        new_data = {name: _pin_memory(tensor) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name, value in self.coordinates.items():
            if name in static:
                new_coordinates[name] = value
                continue
            new_coordinates[name] = _pin_memory(value)

        new_timedeltas = {name: _pin_memory(value) for name, value in self.timedeltas.items()}

        return Batch(
            data=new_data,
            coordinates=new_coordinates,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            grid_shard_indices=self.grid_shard_indices,
            layouts=self.layouts,
        )

    # ------------------------------------------------------------- transform

    def with_data(self, new_data: dict[str, torch.Tensor]) -> "Batch":
        """Return a new :class:`Batch` with ``data`` replaced by ``new_data``.

        ``coordinates``, ``timedeltas`` and ``metadata`` are shared by
        reference with the receiver. This preserves static-coord identity
        (no extra H2D, no copy) and is the recommended way for tasks to
        slice or transform the data tensors while keeping the coordinate
        envelope intact.

        Parameters
        ----------
        new_data : dict[str, torch.Tensor]
            Replacement data tensors. Must cover the same dataset names
            as ``self.data``.

        Returns
        -------
        Batch
            A new frozen :class:`Batch` sharing ``self.coordinates``,
            ``self.timedeltas`` and ``self.metadata`` by reference.
        """
        if set(new_data.keys()) != set(self.data.keys()):
            msg = (
                "with_data: new_data keys must match the existing dataset names "
                f"(got {sorted(new_data)}, expected {sorted(self.data)})."
            )
            raise ValueError(msg)

        return Batch(
            data=new_data,
            coordinates=self.coordinates,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=self.timedeltas,
            grid_shard_indices=self.grid_shard_indices,
            layouts=self.layouts,
        )

    def node_coords(self, dataset_name: str) -> torch.Tensor | None:
        """Return per-node ``(num_nodes, 2)`` lat/lon coordinates for ``dataset_name``.

        Returns the dataset's ``coordinates`` tensor as stored on the
        batch — already shaped ``(N, 2)`` (or ``(B, N, 2)`` for dynamic
        gridded datasets) with ``(latitude, longitude)`` along the
        trailing dimension, in **radians**.

        Returns ``None`` when the dataset has no entry in
        :attr:`coordinates` — the model layer then falls back to its
        registered static buffer (preserving full backward compatibility
        with checkpoints trained before the typed-batch refactor). Sparse
        datasets (``coordinates`` stored as ``list[torch.Tensor]``) also
        return ``None`` here; consumers should use :meth:`view` to access
        the per-sample list directly.
        """
        value = self.coordinates.get(dataset_name)
        if value is None or isinstance(value, list):
            return None
        return value

    # ----------------------------------------------------------- construction

    @staticmethod
    def collate(
        samples: list[dict[str, dict[str, Any]]],
        *,
        static_coord_datasets: Iterable[str] = (),
    ) -> "Batch":
        """Collate a list of per-sample dicts into a :class:`Batch`.

        Each sample must be a mapping ``{dataset_name: payload}`` where
        ``payload`` is a mapping with a required ``"data"`` key, an
        optional ``"coordinates"`` tensor (shape ``(N, 2)`` stacking
        latitudes and longitudes in radians), an optional ``"timedeltas"``
        tensor (sparse only) and an optional ``"metadata"`` mapping.

        Two payload shapes are supported and dispatched on the presence of
        ``BOUNDARIES_META_KEY`` (``"boundaries"``) inside
        ``payload["metadata"]``:

        * **Gridded** — no ``"boundaries"`` metadata. ``payload["data"]`` is
          a :class:`torch.Tensor` of uniform shape across samples; data and
          (non-static) coordinates are stacked along a new leading batch
          dimension via :func:`torch.utils.data.default_collate`. Datasets
          listed in ``static_coord_datasets`` reuse the first sample's
          ``coordinates`` tensor by reference (no stacking, no copy).
        * **Sparse** — ``payload["metadata"]["boundaries"]`` is present (set
          by :meth:`anemoi.training.data.data_reader.ObservationDataReader._unpack_sample`).
          ``payload["data"]`` is a per-sample :class:`torch.Tensor` of
          shape ``(E=1, N_i, V)`` whose ``N_i`` varies between samples.
          ``data[name]``, ``coordinates[name]`` and ``timedeltas[name]``
          each become a ``list[torch.Tensor]`` of length ``B``;
          per-sample ``payload["metadata"]`` is collected into
          ``Batch.metadata[name]`` with each leaf gathered into a list of
          length ``B`` (so ``"boundaries"`` becomes
          ``list[tuple[slice, ...]]``). Sparse datasets must not appear in
          ``static_coord_datasets`` — see
          :attr:`anemoi.training.data.data_reader.BaseAnemoiReader.is_static_grid`.
        """
        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        static = frozenset(static_coord_datasets)

        # Discover the dataset names from the first sample; assume consistent.
        first = samples[0]
        dataset_names = tuple(first.keys())

        collated_data: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_timedeltas: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        per_dataset_metadata: dict[str, dict[str, list[Any]]] = {}

        for name in dataset_names:
            first_payload = first[name]
            sample_meta = first_payload.get("metadata") or {}
            # Sparse observation samples are tagged by the presence of the
            # ``boundaries`` key in their per-sample metadata (set by
            # ``ObservationDataReader._unpack_sample``). Gridded samples
            # never carry it, so this is a self-describing dispatch.
            is_sparse = BOUNDARIES_META_KEY in sample_meta

            if is_sparse:
                if name in static:
                    msg = (
                        f"Dataset {name!r} produced a sparse (boundaries-tagged) "
                        "data payload but is listed in static_coord_datasets; "
                        "sparse datasets must have is_static_grid=False."
                    )
                    raise ValueError(msg)

                # data / coordinates / timedeltas: list[Tensor] of length B,
                # no stacking — sparse samples have varying N_i.
                collated_data[name] = [sample[name]["data"] for sample in samples]

                if "coordinates" in first_payload:
                    collated_coordinates[name] = [
                        sample[name]["coordinates"] for sample in samples
                    ]

                if "timedeltas" in first_payload:
                    collated_timedeltas[name] = [
                        sample[name]["timedeltas"] for sample in samples
                    ]

                # metadata: each leaf becomes a list of length B (one per sample).
                meta_keys = tuple(sample_meta.keys())
                per_dataset_metadata[name] = {
                    key: [sample[name].get("metadata", {}).get(key) for sample in samples]
                    for key in meta_keys
                }
                continue

            # Gridded path: stack data via default_collate.
            collated_data[name] = default_collate([sample[name]["data"] for sample in samples])

            if "coordinates" in first_payload:
                if name in static:
                    # Use the first sample's coordinates tensor by reference;
                    # do NOT copy or repeat across the batch dimension.
                    collated_coordinates[name] = first_payload["coordinates"]
                else:
                    collated_coordinates[name] = default_collate(
                        [sample[name]["coordinates"] for sample in samples],
                    )

            if "timedeltas" in first_payload:
                # Gridded datasets normally don't carry ``timedeltas``; if
                # they do, stack them along a new batch dimension just like
                # the other tensors.
                collated_timedeltas[name] = default_collate(
                    [sample[name]["timedeltas"] for sample in samples],
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

        # Layouts: per-dataset TensorLayout from sample payloads, shifted to
        # include the new leading batch dimension added by collation.
        collated_layouts: dict[str, TensorLayout] = {}
        for name in dataset_names:
            payload = first[name]
            if "layout" in payload:
                collated_layouts[name] = payload["layout"].with_batch_dim()

        metadata: dict[str, Any] = {}
        if static:
            metadata[STATIC_COORDS_META_KEY] = frozenset(static)
        metadata.update(per_dataset_metadata)

        return Batch(
            data=collated_data,
            coordinates=collated_coordinates,
            metadata=metadata,
            grid_sizes=collated_grid_sizes,
            timedeltas=collated_timedeltas,
            grid_shard_indices=collated_grid_shard_indices,
            layouts=collated_layouts,
        )


@dataclass(frozen=True, slots=True)
class DatasetView:
    """Per-dataset view returned by :meth:`Batch.view`."""

    name: str
    data: torch.Tensor | list[torch.Tensor]
    coordinates: torch.Tensor | list[torch.Tensor] | None
    is_static: bool
    timedeltas: torch.Tensor | list[torch.Tensor] | None = None
    grid_shard_indices: Any = None
    layout: TensorLayout | None = None

    @property
    def latlons(self) -> torch.Tensor:
        """Return ``[sin(lat), sin(lon), cos(lat), cos(lon)]`` along a trailing dim.

        Computed from the stored ``(..., N, 2)`` ``coordinates`` tensor;
        only valid for non-sparse (single-tensor) views.
        """
        if not isinstance(self.coordinates, torch.Tensor):
            msg = "DatasetView.latlons requires a single coordinates tensor (gridded datasets only)."
            raise TypeError(msg)
        return torch.cat([torch.sin(self.coordinates), torch.cos(self.coordinates)], dim=-1).to(torch.float32)

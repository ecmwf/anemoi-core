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
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace as dataclass_replace
from typing import Any

import einops
import torch
from torch.utils.data import default_collate
from anemoi.models.data.tensor_layout import TensorLayout

# Key in ``Batch.metadata`` listing dataset names whose coordinate tensors are
# static (allocated once, shared by reference, not transferred per step).
STATIC_COORDS_META_KEY = "static_coords"

# Reserved per-dataset metadata key carrying sparse-obs per-time boundaries
# as a ``list[tuple[slice, ...]]`` (one entry per batch sample). Untouched by
# device transfer / pinning since ``slice`` is not a tensor.
BOUNDARIES_META_KEY = "boundaries"


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
        """Return a per-dataset view bundling data, coordinates, layout and metadata."""
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        layout = self.layouts.get(dataset_name)
        if layout is None:
            msg = (
                f"Dataset {dataset_name!r} has no TensorLayout in this batch; "
                "layout is required for view-based access. "
                "Set batch.layouts[name] or pass 'layout' in the sample payload during collation."
            )
            raise ValueError(msg)
        per_dataset_meta = self.metadata.get(dataset_name) if isinstance(self.metadata.get(dataset_name), dict) else None
        boundaries = per_dataset_meta.get(BOUNDARIES_META_KEY) if per_dataset_meta else None
        return DatasetView(
            name=dataset_name,
            data=self.data[dataset_name],
            coordinates=self.coordinates.get(dataset_name),
            is_static=self.is_static_coords(dataset_name),
            timedeltas=self.timedeltas.get(dataset_name),
            grid_shard_indices=self.grid_shard_indices.get(dataset_name),
            layout=layout,
            boundaries=boundaries,
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

    # ------------------------------------------------------- bulk indexing

    def select_time(self, indices: "slice | Sequence[int] | int") -> "Batch":
        """Return a new :class:`Batch` with each dataset restricted to ``indices``.

        Delegates per dataset to :meth:`DatasetView.select_time`, which
        dispatches on ``layout.time_in_grid`` to handle gridded and
        sparse observation datasets uniformly.
        """
        new_data: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        new_coords: dict[str, torch.Tensor | list[torch.Tensor]] = dict(self.coordinates)
        new_timedeltas: dict[str, torch.Tensor | list[torch.Tensor]] = dict(self.timedeltas)
        new_metadata = dict(self.metadata)

        for name in self.dataset_names:
            view = self.view(name).select_time(indices)
            new_data[name] = view.data
            if view.coordinates is not None and name in self.coordinates:
                new_coords[name] = view.coordinates
            if view.timedeltas is not None and name in self.timedeltas:
                new_timedeltas[name] = view.timedeltas
            if view.boundaries is not None:
                # Mirror the updated boundaries back into per-dataset metadata.
                per_ds = dict(new_metadata.get(name, {})) if isinstance(new_metadata.get(name), dict) else {}
                per_ds[BOUNDARIES_META_KEY] = view.boundaries
                new_metadata[name] = per_ds

        return Batch(
            data=new_data,
            coordinates=new_coords,
            metadata=new_metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            grid_shard_indices=self.grid_shard_indices,
            layouts=self.layouts,
        )

    def select_vars(self, indices_per_dataset: dict[str, Sequence[int] | torch.Tensor | slice]) -> "Batch":
        """Return a new :class:`Batch` with per-dataset variable indexing applied.

        Datasets absent from ``indices_per_dataset`` pass through unchanged.
        """
        new_data: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name in self.dataset_names:
            if name in indices_per_dataset:
                new_data[name] = self.view(name).select_vars(indices_per_dataset[name]).data
            else:
                new_data[name] = self.data[name]
        return self.with_data(new_data)

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
            if BOUNDARIES_META_KEY in sample_meta:
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
    """Per-dataset view returned by :meth:`Batch.view`.

    Bundles the per-dataset payload (data, coordinates, timedeltas) with
    its :class:`TensorLayout` so callers can index logical axes (``time``,
    ``variables``) without hard-coded dimension positions. The same API
    works for gridded and sparse observation datasets thanks to the
    ``layout.time_in_grid`` dispatch.
    """

    name: str
    data: torch.Tensor | list[torch.Tensor]
    coordinates: torch.Tensor | list[torch.Tensor] | None
    is_static: bool
    layout: TensorLayout
    timedeltas: torch.Tensor | list[torch.Tensor] | None = None
    grid_shard_indices: Any = None
    boundaries: list[tuple[slice, ...]] | None = None

    # ------------------------------------------------------------- helpers

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

    def time_slices(self) -> list[tuple[slice, ...]] | None:
        """Return per-sample boundary slices for sparse views; ``None`` otherwise.

        For sparse observation datasets (``layout.time_in_grid=True``) the
        ``boundaries`` describe which contiguous ranges of the grid axis
        belong to each time step; for gridded datasets time is an explicit
        dim so this returns ``None``.
        """
        return self.boundaries if self.layout.time_in_grid else None

    # ------------------------------------------------------- core indexing

    def select_time(self, indices: "slice | Sequence[int] | int") -> "DatasetView":
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
        DatasetView
            A new view with the same :class:`TensorLayout` but reduced
            time extent.
        """
        if isinstance(indices, slice):
            time_size = self._time_axis_size()
            idx_list = list(range(*indices.indices(time_size)))
        elif isinstance(indices, int):
            idx_list = [int(indices)]
        else:
            idx_list = [int(i) for i in indices]

        if not self.layout.time_in_grid:
            return self._select_time_gridded(idx_list)

        return self._select_time_sparse(idx_list)

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

    def _select_time_gridded(self, indices: list[int]) -> "DatasetView":
        if self.layout.time is None:
            msg = f"Layout {self.layout!r} has no time axis; cannot select_time on a gridded view."
            raise ValueError(msg)
        assert isinstance(self.data, torch.Tensor), "Gridded view must wrap a single tensor."
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.data.device)
        new_data = self.data.index_select(self.layout.time, idx)
        return dataclass_replace(self, data=new_data)

    def _select_time_sparse(self, indices: list[int]) -> "DatasetView":
        if self.boundaries is None:
            msg = "Sparse view has no 'boundaries' metadata; cannot select_time."
            raise ValueError(msg)
        assert isinstance(self.data, list), "Sparse view must wrap a list[Tensor]."

        coords_list = self.coordinates if isinstance(self.coordinates, list) else None
        if self.coordinates is not None and coords_list is None:
            msg = "Sparse view coordinates must be a list[Tensor]."
            raise TypeError(msg)
        td_list = self.timedeltas if isinstance(self.timedeltas, list) else None
        if self.timedeltas is not None and td_list is None:
            msg = "Sparse view timedeltas must be a list[Tensor]."
            raise TypeError(msg)

        grid_dim = self.layout.grid

        new_data: list[torch.Tensor] = []
        new_coords: list[torch.Tensor] | None = [] if coords_list is not None else None
        new_timedeltas: list[torch.Tensor] | None = [] if td_list is not None else None
        new_boundaries: list[tuple[slice, ...]] = []

        for sample_idx, sample_bounds in enumerate(self.boundaries):
            selected_slices = [sample_bounds[t] for t in indices]
            sample_data = self.data[sample_idx]
            data_pieces = [sample_data.narrow(grid_dim, s.start, s.stop - s.start) for s in selected_slices]
            new_data.append(torch.cat(data_pieces, dim=grid_dim) if data_pieces else sample_data.narrow(grid_dim, 0, 0))

            if new_coords is not None and coords_list is not None:
                sample_coords = coords_list[sample_idx]
                coord_pieces = [sample_coords[s.start : s.stop] for s in selected_slices]
                new_coords.append(
                    torch.cat(coord_pieces, dim=0) if coord_pieces else sample_coords[:0],
                )

            if new_timedeltas is not None and td_list is not None:
                sample_td = td_list[sample_idx]
                td_pieces = [sample_td[s.start : s.stop] for s in selected_slices]
                new_timedeltas.append(
                    torch.cat(td_pieces, dim=0) if td_pieces else sample_td[:0],
                )

            offset = 0
            compact: list[slice] = []
            for s in selected_slices:
                length = s.stop - s.start
                compact.append(slice(offset, offset + length))
                offset += length
            new_boundaries.append(tuple(compact))

        return dataclass_replace(
            self,
            data=new_data,
            coordinates=new_coords if new_coords is not None else self.coordinates,
            timedeltas=new_timedeltas if new_timedeltas is not None else self.timedeltas,
            boundaries=new_boundaries,
        )

    def select_vars(self, indices: Sequence[int] | torch.Tensor | slice) -> "DatasetView":
        """Return a new view restricted to the given variable indices.

        Indexes along ``layout.variables`` for both gridded and sparse
        datasets. Coordinates / timedeltas / boundaries are unchanged.
        """
        if isinstance(self.data, list):
            new_data = [self._index_vars(t, indices) for t in self.data]
        else:
            new_data = self._index_vars(self.data, indices)
        return dataclass_replace(self, data=new_data)

    def _index_vars(self, tensor: torch.Tensor, indices: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        var_dim = self.layout.axis("variables", ndim=tensor.ndim)
        if isinstance(indices, slice):
            slicer: list[Any] = [slice(None)] * tensor.ndim
            slicer[var_dim] = indices
            return tensor[tuple(slicer)]
        idx = torch.as_tensor(list(indices) if not isinstance(indices, torch.Tensor) else indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(var_dim, idx)

    # ------------------------------------------------------- model adapters

    def rearrange_for_encoder(self) -> torch.Tensor:
        """Flatten the view's data to the encoder's expected ``(*, T*V)`` layout.

        Returns
        -------
        torch.Tensor
            For gridded views: shape ``(batch * ensemble * grid, time * variables)``.
            For sparse views: shape ``(ensemble * total_grid, variables)``
            obtained by concatenating per-sample tensors along the grid axis.
        """
        if not self.layout.time_in_grid:
            assert isinstance(self.data, torch.Tensor), "Gridded view must wrap a single tensor."
            return einops.rearrange(
                self.data,
                "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
            )
        # Sparse: list[Tensor] of shape (ensemble=1, N_i, V). Concatenate
        # samples along the grid axis to produce a single flat tensor.
        assert isinstance(self.data, list), "Sparse view must wrap a list[Tensor]."
        flat = torch.cat(self.data, dim=self.layout.grid)
        # Result shape (ensemble, total_N, V) -> rearrange to (ensemble*total_N, V)
        return einops.rearrange(flat, "ensemble grid vars -> (ensemble grid) vars")

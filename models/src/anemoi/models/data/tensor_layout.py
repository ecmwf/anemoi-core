# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """Maps logical axes to physical dimension positions.

    Describes the semantic meaning of each dimension in a per-dataset data
    tensor so that downstream code (tasks, losses, metrics) can index axes
    by name rather than by hard-coded position.

    Parameters
    ----------
    batch : int or None
        Position of the batch dimension (``None`` before collation, or for
        sparse datasets where the batch is the outer ``list[Tensor]``).
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

    Notes
    -----
    For sparse observation datasets (``time_in_grid=True``) the per-sample
    inner tensor has shape ``(ensemble, grid, variables)`` and the batch
    dimension is represented by the outer Python list (one tensor per
    sample). ``batch`` therefore stays ``None`` even after collation.
    """

    batch: int | None = None
    time: int | None = None
    ensemble: int | None = None
    grid: int = -2
    variables: int = -1
    time_in_grid: bool = False

    _AXIS = ("batch", "time", "ensemble", "grid", "variables")

    @property
    def dims(self) -> set[str]:
        """Set of logical axes defined by this layout."""
        return {name for name in self._AXIS if getattr(self, name) is not None}

    @property
    def ndims(self) -> int:
        """Number of dimensions in tensors with this layout."""
        return len(self.dims)

    @property
    def pattern(self) -> str:
        """Einops pattern string for this layout, with named axes."""
        parts = list(sorted(self.dims, key=lambda x: getattr(self, x)))
        return " ".join(parts)

    def with_batch_dim(self) -> "TensorLayout":
        """Return a new layout shifted by +1 to account for a leading batch dim.

        For sparse datasets (``time_in_grid=True``) the batch is
        represented by the outer ``list[Tensor]`` rather than a tensor
        dimension; the layout is returned unchanged.
        """
        if self.batch is not None or self.time_in_grid:
            return self

        return TensorLayout(
            batch=0,
            time=self.time + 1 if self.time is not None else None,
            ensemble=self.ensemble + 1 if self.ensemble is not None else None,
            grid=self.grid + 1 if self.grid >= 0 else self.grid,
            variables=self.variables + 1 if self.variables >= 0 else self.variables,
            time_in_grid=self.time_in_grid,
        )

    def without_batch_dim(self) -> "TensorLayout":
        """Return a new layout with the batch dim removed (inverse of :meth:`with_batch_dim`).

        Positive non-``None`` axis positions are shifted by ``-1``; negative
        positions are left unchanged.
        """
        if self.batch is None:
            return self
        return TensorLayout(
            batch=None,
            time=self.time - 1 if self.time is not None and self.time > 0 else self.time,
            ensemble=self.ensemble - 1 if self.ensemble is not None and self.ensemble > 0 else self.ensemble,
            grid=self.grid - 1 if self.grid > 0 else self.grid,
            variables=self.variables - 1 if self.variables > 0 else self.variables,
            time_in_grid=self.time_in_grid,
        )

    def axis(self, name: str, *, ndim: int | None = None) -> int:
        """Return the physical dim index for logical axis ``name``.

        Negative indices are normalised against ``ndim`` when provided.
        Raises :class:`ValueError` if the requested axis is not defined for
        this layout (e.g. ``time`` on a ``time_in_grid=True`` layout).
        """
        pos = getattr(self, name, None)
        if pos is None:
            msg = f"Logical axis {name!r} is not defined for this layout: {self!r}"
            raise ValueError(msg)
        if pos < 0 and ndim is not None:
            pos = pos + ndim
        return pos

    def has_axis(self, name: str) -> bool:
        """Return whether the layout defines a position for logical axis ``name``."""
        return getattr(self, name, None) is not None

    def __repr__(self) -> str:
        parts = []
        for name in ("batch", "time", "ensemble", "grid", "variables"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        if self.time_in_grid:
            parts.append("time_in_grid=True")
        return f"TensorLayout({', '.join(parts)})"

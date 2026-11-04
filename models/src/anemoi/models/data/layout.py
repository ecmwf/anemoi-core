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
        tabular datasets where the batch is the outer ``list[Tensor]``).
    time : int or None
        Position of the explicit time dimension. ``None`` for tabular datasets,
        whose time windows are stacked along the grid axis.
    ensemble : int or None
        Position of the ensemble dimension (``None`` when absent).
    grid : int
        Position of the grid / spatial-points dimension. Defaults to the second-last axis.
    variables : int
        Position of the variable (channel) dimension. Defaults to the last axis.

    A layout names every axis of its tensors, so its rank is the number of axes it
    defines. Negative positions count from the end of that rank, as in Python indexing,
    and are resolved on construction: the stored positions are always non-negative, so
    ``TensorLayout(time=0, ensemble=1)`` equals ``TensorLayout(time=0, ensemble=1, grid=2,
    variables=3)``.

    Notes
    -----
    A layout only describes the axes of a tensor, not the kind of data. Whether a
    dataset is gridded or tabular is given by its sample and source classes (see
    :mod:`anemoi.models.data.sample`), which also decide how the layout changes on
    collation.
    """

    batch: int | None = None
    time: int | None = None
    ensemble: int | None = None
    grid: int = -2
    variables: int = -1

    #: Logical axis names, in canonical order.
    AXES = ("batch", "time", "ensemble", "grid", "variables")

    def __post_init__(self) -> None:
        """Resolve negative positions against the layout's rank, and check the axes.

        A layout names every axis of its tensors, so its rank is the number of axes it
        defines and every position is known on construction. Storing them non-negative
        makes equivalent layouts equal and keeps :attr:`pattern` in physical order.
        """
        ndim = self.ndim
        positions = {}
        for name in self.AXES:
            pos = getattr(self, name)
            if pos is None:
                continue
            if not -ndim <= pos < ndim:
                msg = f"TensorLayout axis {name}={pos} is out of range for a layout of {ndim} axes."
                raise ValueError(msg)
            if pos < 0:
                pos += ndim
                object.__setattr__(self, name, pos)
            positions[name] = pos
        if len(set(positions.values())) != len(positions):
            msg = f"TensorLayout places several axes at the same position: {positions}."
            raise ValueError(msg)

    @classmethod
    def from_tuple(cls, *args) -> "TensorLayout":
        """Create a TensorLayout from a tuple of axis names.

        Parameters
        ----------
        *args : str
            Logical axis names in the order they appear in the tensor.

        Returns
        -------
        TensorLayout
            Layout with each named axis at its supplied position.
        """
        axis_positions = {name: i for i, name in enumerate(args)}
        return cls(
            batch=axis_positions.get("batch"),
            time=axis_positions.get("time"),
            ensemble=axis_positions.get("ensemble"),
            grid=axis_positions.get("grid", -2),
            variables=axis_positions.get("variables", -1),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Logical axis names of this layout, ordered by physical position."""
        layout = self.normalized(self.ndim)
        return tuple(sorted(layout.dims, key=lambda name: getattr(layout, name)))

    @property
    def dims(self) -> set[str]:
        """Set of logical axes defined by this layout."""
        return {name for name in self.AXES if getattr(self, name) is not None}

    @property
    def ndim(self) -> int:
        """Number of dimensions in tensors with this layout."""
        return len(self.dims)

    @property
    def pattern(self) -> str:
        """Einops pattern string for this layout, with named axes."""
        parts = list(sorted(self.dims, key=lambda x: getattr(self, x)))
        return " ".join(parts)

    def with_batch_dim(self) -> "TensorLayout":
        """Return a new layout with a leading batch axis, every other axis shifted by +1.

        Returns the layout unchanged when it already has a batch axis.
        """
        if self.batch is not None:
            return self

        shifted = {name: getattr(self, name) + 1 for name in self.dims}
        return TensorLayout(batch=0, **shifted)

    def without_batch_dim(self) -> "TensorLayout":
        """Return a new layout with the batch axis removed (inverse of :meth:`with_batch_dim`).

        The axes after the batch axis move one position forward.
        """
        if self.batch is None:
            return self
        remaining = {name: getattr(self, name) for name in self.dims - {"batch"}}
        return TensorLayout(**{name: pos - 1 if pos > self.batch else pos for name, pos in remaining.items()})

    def axis(self, name: str, *, ndim: int | None = None) -> int:
        """Return the physical dim index for logical axis ``name``.

        Positions are stored non-negative, so ``ndim`` only matters for a layout
        unpickled from before they were resolved on construction.

        Raises
        ------
        :class:`ValueError`
            if the requested axis is not defined for this layout (e.g.
            ``time`` on a tabular layout).
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

    def normalized(self, ndim: int) -> "TensorLayout":
        """Return a layout with physical axes resolved against the tensor rank."""
        positions = {name: self.axis(name, ndim=ndim) for name in self.AXES if self.has_axis(name)}
        if len(positions) != ndim or set(positions.values()) != set(range(ndim)):
            msg = f"Layout {self!r} must describe each of the {ndim} tensor axes exactly once."
            raise ValueError(msg)
        return TensorLayout(**positions)

    def __repr__(self) -> str:
        parts = []
        for name in self.AXES:
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        return f"TensorLayout({', '.join(parts)})"

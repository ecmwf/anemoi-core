# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import uuid
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Self

import torch
from torch import nn

from anemoi.models.data import TensorLayout
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def reshape_scaler(dims: tuple[str, ...], scaler: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
    """Reshapes a scaler tensor to align with specific logical axes of a target layout for broadcasting.

    The scaler's axes follow ``dims``, which need not be in the layout's order: they are permuted
    into that order first, so e.g. a ``(variables, grid)`` scaler lines up with a grid-first layout.
    """
    target_ndim = layout.ndim
    physical_dims = [layout.axis(dim_name, ndim=target_ndim) for dim_name in dims]
    order = sorted(range(len(dims)), key=physical_dims.__getitem__)

    new_shape = [1] * target_ndim
    for i in order:
        new_shape[physical_dims[i]] = scaler.shape[i]

    return scaler.permute(order).reshape(new_shape)


def as_dimension_tuple(dimensions: int | str | Sequence[int | str]) -> tuple[int | str, ...]:
    """Normalise a dimension selector to a tuple of dimension identifiers.

    A single dimension may be given as a bare axis number or as a bare `TensorDim`.
    Wrapping both scalar forms keeps name-based and axis-based selection behaving the same.

    Parameters
    ----------
    dimensions : int | str | Sequence[int | str]
        One dimension, or a sequence of them.

    Returns
    -------
    tuple[int | str, ...]
        The dimensions as a tuple.
    """
    if isinstance(dimensions, (int, str)):
        return (dimensions,)
    return tuple(dimensions)


TENSOR_SPEC = tuple[int | tuple[int, ...], torch.Tensor]
"""Scale Tensor specification type.

A tuple of (dimension, tensor) where:
- dimension can be a single int or a tuple of ints specifying the dimensions the tensor should be applied to.
- tensor is a torch.Tensor that will be applied to the specified dimensions.
"""


class Shape:
    """Shape resolving object."""

    def __init__(self, func: Callable[[int], int]):
        self.func = func

    def __getitem__(self, dimension: int) -> int:
        return self.func(dimension)


class ScaleTensor(nn.Module):
    """Dynamically resolved tensor scaling class.

    Allows a user to specify a scaler and the dimensions it should be applied to.
    The class will then enforce that additional scalers are compatible with the specified dimensions.

    Dimensions are named (`TensorDim`), and `scale_iteratively` multiplies the input tensor by
    every scaler, aligned to the input's axes through a `TensorLayout`.

    Additionally, the class can be subsetted to only return a subset of the scalers, but
    only from those given names.

    Examples
    --------
    >>> layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    >>> tensor = torch.randn(2, 1, 1, 4, 5)
    >>> scalers = ScaleTensor(grid=(TensorDim.GRID, torch.randn(4)))
    >>> scalers.add_scaler(TensorDim.VARIABLE, torch.randn(5), name="variable")
    >>> scalers.scale_iteratively(tensor, layout=layout).shape
    torch.Size([2, 1, 1, 4, 5])
    """

    _tensors: dict[str, TENSOR_SPEC]

    def __init__(
        self,
        scalers: dict[str, TENSOR_SPEC] | TENSOR_SPEC | None = None,
        *tensors: TENSOR_SPEC,
        **named_tensors: dict[str, TENSOR_SPEC],
    ):
        """ScaleTensor constructor.

        Parameters
        ----------
        scalers : dict[str, TENSOR_SPEC] | TENSOR_SPEC | None, optional
            Scalers to initalise with, by default None
        tensors : TENSOR_SPEC
            Args form of (dimension, tensor) to add to the scalers
            Will be given a random uuid name
        named_tensors : dict[str, TENSOR_SPEC]
            Kwargs form of {name: (dimension, tensor)} to add to the scalers
        """
        super().__init__()

        self._tensors = {}

        named_tensors.update(scalers or {})
        self.add(named_tensors)

        for tensor_spec in tensors:
            self.add_scaler(*tensor_spec)

    @property
    def tensors(self) -> dict[str, TENSOR_SPEC]:
        """Get the scalers as a dictionary of name to (dimension, tensor) pairs."""
        return dict(self._tensors)

    def __getattr__(self, name: str) -> Any:
        """Expose scalers as attributes stored in self._tensors."""
        tensors = self.__dict__.get("_tensors")
        if tensors is not None and name in tensors:
            return tensors[name][1]
        return super().__getattr__(name)

    def _apply(self, fn: Callable, recurse: bool = True) -> Self:
        """Apply device and dtype conversions to the scalers.

        ``nn.Module._apply`` handles parameters and registered buffers. The scaler
        tensors (``_tensors``) are deliberately neither, so we apply the same
        conversion to them explicitly.
        """
        super()._apply(fn, recurse=recurse)
        for name, (dimension, scaler) in self._tensors.items():
            self._tensors[name] = (dimension, fn(scaler))
        return self

    @property
    def specified_dimensions(self) -> dict[str, tuple[int]]:
        """Get the specified dimensions for each scaler."""
        return {key: dim for key, (dim, _) in self._tensors.items()}

    @property
    def shape(self) -> Shape:
        """Get the shape of the scale tensor.

        Returns a Shape object to be indexed,
        Will only resolve those dimensions specified in the tensors.
        """

        def get_dim_shape(dimension: int) -> int:
            for dim_assign, tensor in self.tensors.values():
                if isinstance(dim_assign, tuple) and dimension in dim_assign:
                    return tensor.shape[list(dim_assign).index(dimension)]

            unique_dims = {dim for dim_assign in self.specified_dimensions.values() for dim in dim_assign}
            error_msg = (
                f"Could not find shape of dimension {dimension}. "
                f"Tensors are only specified for dimensions {list(unique_dims)}."
            )
            raise IndexError(error_msg)

        return Shape(get_dim_shape)

    def has_scaler_for_dim(self, dim: TensorDim) -> bool:
        """Check if there is a scaler for the given dimension."""
        return len(self.subset_by_dim(dim.value).tensors) > 0

    def has_dim(self, dimension: int | str) -> bool:
        """Whether any scaler applies along ``dimension`` (an axis number or a `TensorDim`)."""
        return any(dimension in dims for dims, _ in self._tensors.values())

    def validate_scaler(self, dimension: int | str | tuple[int | str, ...], scaler: torch.Tensor) -> None:
        """Check if the scaler is compatible with the given dimension.

        Parameters
        ----------
        dimension : int | str | tuple[int | str, ...]
            Dimensions to check `scaler` against: an axis number, a `TensorDim`, or a tuple of them
        scaler : torch.Tensor
            Scaler tensor to check

        Raises
        ------
        ValueError
            If the scaler is not compatible with the given dimension
        """
        for scaler_dim, dim in enumerate(as_dimension_tuple(dimension)):
            if not self.has_dim(dim) or scaler.shape[scaler_dim] == 1 or self.shape[dim] == 1 or dim == TensorDim.GRID:
                continue

            if self.shape[dim] != scaler.shape[scaler_dim]:
                error_msg = (
                    f"Incoming scaler shape {scaler.shape} at dimension {scaler_dim} "
                    f"does not match shape of saved scaler. Expected {self.shape[dim]}"
                )
                raise ValueError(error_msg)

    def add_scaler(
        self,
        dimension: int | tuple[int],
        scaler: torch.Tensor,
        *,
        name: str | None = None,
    ) -> Self:
        """Add new scaler to be applied along `dimension`.

        Dimension can be a single int even for a multi-dimensional scaler,
        in this case the dimensions are assigned as a range starting from the given int.
        Negative indexes are also valid, and will be resolved against the tensor's ndim.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimension/s to apply the scaler to
        scaler : torch.Tensor
            Scaler tensor to apply
        name : str | None, optional
            Name of the scaler, by default None

        Returns
        -------
        ScaleTensor
            ScaleTensor with the scaler removed
        """
        if not isinstance(scaler, torch.Tensor):
            scaler = torch.tensor([scaler]) if isinstance(scaler, int | float) else torch.tensor(scaler)
        assert (
            not scaler.requires_grad
        ), f"Scaler tensors must not require gradients. Got requires_grad=True for scaler {name!r}."

        if isinstance(dimension, str):
            # A bare `TensorDim` names one dimension; `tuple()` of it would yield its characters.
            dimension = (dimension,)
        elif isinstance(dimension, int):
            if len(scaler.shape) == 1:
                dimension = (dimension,)
            else:
                dimension = tuple(dimension + i for i in range(len(scaler.shape)))
        else:
            dimension = tuple(dimension)

        if name is None:
            name = str(uuid.uuid4())

        if name in self._tensors:
            msg = f"Scaler {name!r} already exists in scalers."
            raise ValueError(msg)

        try:
            self.validate_scaler(dimension, scaler)
        except ValueError as e:
            error_msg = f"Validating tensor {name!r} raised an error."
            raise ValueError(error_msg) from e

        self._tensors[name] = (dimension, scaler)

        return self

    def remove_scaler(self, scaler_to_remove: str | int) -> Self:
        """Remove scaler from ScaleTensor.

        Parameters
        ----------
        scaler_to_remove : str | int
            Name or index of tensor to remove

        Raises
        ------
        ValueError
            If the scaler is not in the scalers

        Returns
        -------
        ScaleTensor
            ScaleTensor with the scaler removed
        """
        for scaler_to_pop in self.subset(scaler_to_remove).tensors:
            self._tensors.pop(scaler_to_pop)
        return self

    def freeze_state(self) -> "FrozenStateRecord":  # noqa: F821
        """Freeze the state of the scaler with a context manager.

        Any changes made will be reverted on exit.

        Returns
        -------
        FrozenStateRecord
            Context manager to freeze the state of this ScaleTensor
        """
        record_of_scalers: dict = self.tensors.copy()

        class FrozenStateRecord:
            """Freeze the state of the ScaleTensor. Any changes will be reverted on exit."""

            def __enter__(self):
                pass

            def __exit__(context_self, *a):  # noqa: N805
                for key in list(self._tensors.keys()):
                    if key not in record_of_scalers:
                        self.remove_scaler(key)

                for key in record_of_scalers:
                    if key not in self:
                        self.add_scaler(*record_of_scalers[key], name=key)

        return FrozenStateRecord()

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        """Update an existing scaler maintaining original dimensions.

        If `override` is False, the scaler must be valid against the original dimensions.
        If `override` is True, the scaler will be updated regardless of validity against original scaler.

        Parameters
        ----------
        name : str
            Name of the scaler to update
        scaler : torch.Tensor
            New scaler tensor
        override : bool, optional
            Whether to override the scaler ignoring dimension compatibility, by default False
        """
        if not isinstance(scaler, torch.Tensor):
            scaler = torch.tensor([scaler]) if isinstance(scaler, int | float) else torch.tensor(scaler)
        assert (
            not scaler.requires_grad
        ), f"Scaler tensors must not require gradients. Got requires_grad=True for scaler {name!r}."

        if name not in self._tensors:
            msg = f"scaler {name!r} not found in scalers."
            raise ValueError(msg)

        dimension = self._tensors[name][0]

        if not override:
            # Multiple updating scalers sharing a dynamic dimension are not supported;
            # other updating scalers would still expose their previous shape here.
            excluded_scaler = self._tensors.pop(name)
            try:
                self.validate_scaler(dimension, scaler)
            finally:
                self._tensors[name] = excluded_scaler

        self._tensors[name] = (dimension, scaler)

    def add(self, new_scalers: dict[str, TENSOR_SPEC] | list[TENSOR_SPEC] | None = None, **kwargs) -> None:
        """Add multiple scalers to the existing scalers.

        Parameters
        ----------
        new_scalers : dict[str, TENSOR_SPEC] | list[TENSOR_SPEC] | None, optional
            Scalers to add, see `add_scaler` for more info, by default None
        **kwargs:
            Kwargs form of {name: (dimension, tensor)} to add to the scalers
        """
        if isinstance(new_scalers, list):
            for tensor_spec in new_scalers:
                self.add_scaler(*tensor_spec)
        else:
            kwargs.update(new_scalers or {})
        for name, tensor_spec in kwargs.items():
            self.add_scaler(*tensor_spec, name=name)

    def update(self, updated_scalers: dict[str, torch.Tensor] | None = None, override: bool = False, **kwargs) -> None:
        """Update multiple scalers in the existing scalers.

        If `override` is False, the scaler must be valid against the original dimensions.
        If `override` is True, the scaler will be updated regardless of shape.

        Parameters
        ----------
        updated_scalers : dict[str, torch.Tensor] | None, optional
            Scalers to update, referenced by name, by default None
        override : bool, optional
            Whether to override the scaler ignoring dimension compatibility, by default False
        **kwargs:
            Kwargs form of {name: tensor} to update in the scalers
        """
        kwargs.update(updated_scalers or {})
        for name, tensor in kwargs.items():
            self.update_scaler(name, tensor, override=override)

    def subset(self, scaler_identifier: str | Sequence[str] | int | Sequence[int]) -> Self:
        """Get subset of the scalers, filtering by name or dimension.

        Parameters
        ----------
        scaler_identifier : str | Sequence[str] | int | Sequence[int]
            Name/s or dimension/s of the scalers to get. Axis numbers and `TensorDim`
            members are dimensions; any other string is a scaler name.

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scaler_identifier, str | int):
            scaler_identifier = [scaler_identifier]
        if any(isinstance(scaler, int | TensorDim) for scaler in scaler_identifier):
            return self.subset_by_dim(scaler_identifier)
        return self.subset_by_str(scaler_identifier)

    def subset_by_str(self, scalers: str | Sequence[str]) -> Self:
        """Get subset of the scalers, filtering by name.

        See `.subset_by_dim` for subsetting by affected dimensions.

        Parameters
        ----------
        scalers : str | Sequence[str]
            Name/s of the scalers to get

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalers, str):
            scalers = [scalers]
        return ScaleTensor(**{name: self.tensors[name] for name in scalers})

    def subset_by_dim(self, dimensions: int | str | Sequence[int | str]) -> Self:
        """Get subset of the scalers, filtering by dimension.

        See `.subset` for subsetting by name.

        Parameters
        ----------
        dimensions : int | str | Sequence[int | str]
            Dimensions to get scalers of, as axis numbers or `TensorDim` names

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalers: dict[str, TENSOR_SPEC] = {}

        dimensions = as_dimension_tuple(dimensions)

        for name, (dim, scaler) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) > 0:
                subset_scalers[name] = (dim, scaler)

        return ScaleTensor(**subset_scalers)

    def without(self, scaler_identifier: str | Sequence[str] | int | Sequence[int]) -> Self:
        """Get subset of the scalers, filtering out by name or dimension.

        Parameters
        ----------
        scaler_identifier : str | Sequence[str] | int | Sequence[int]
            Name/s or dimension/s of the scalers to exclude. Axis numbers and
            `TensorDim` members are dimensions; any other string is a scaler name.

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scaler_identifier, str | int):
            scaler_identifier = [scaler_identifier]
        # TensorDim is a StrEnum, so it must be told apart from a scaler name by type.
        dimensions = [s for s in scaler_identifier if isinstance(s, int | TensorDim)]
        names = [s for s in scaler_identifier if not isinstance(s, int | TensorDim)]
        subset = self.without_by_dim(dimensions) if dimensions else self
        return subset.without_by_str(names) if names else subset

    def without_by_str(self, scalers: str | Sequence[str]) -> Self:
        """Get subset of the scalers, filtering out by name.

        Parameters
        ----------
        scalers : str | Sequence[str]
            Name/s of the scalers to exclude

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalers, str):
            scalers = [scalers]
        return ScaleTensor(**{name: tensor for name, tensor in self.tensors.items() if name not in scalers})

    def without_by_dim(self, dimensions: int | str | Sequence[int | str]) -> Self:
        """Get subset of the scalers, filtering out by dimension.

        Parameters
        ----------
        dimensions : int | str | Sequence[int | str]
            Dimensions to exclude scalers of, as axis numbers or `TensorDim` names

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalers: dict[str, TENSOR_SPEC] = {}

        dimensions = as_dimension_tuple(dimensions)

        for name, (dim, scaler) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) == 0:
                subset_scalers[name] = (dim, scaler)

        return ScaleTensor(**subset_scalers)

    def scale_iteratively(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...] | None = None,
        layout: TensorLayout | None = None,
        *,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Apply the scalers iteratively to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to scale
        subset_indices : tuple[int, ...] | None, optional
            Indices to subset the input tensor, by default None
        layout : TensorLayout | None, optional
            Layout describing the logical axes of ``x``, used to align scalers
            for broadcasting, by default None
        grid_shard_slice : slice | None, optional
            Slice to apply to the grid dimension, by default None

        Returns
        -------
        torch.Tensor
            Scaled tensor.
        """
        if subset_indices is not None and not isinstance(subset_indices, tuple):
            msg = "subset_indices must be a tuple of per-dimension indexers, e.g. (..., indices)"
            raise TypeError(msg)

        # Determine the subset of the input tensor to work with
        x_subset = x[subset_indices] if subset_indices is not None and subset_indices != (...,) else x

        out = x_subset.clone()

        for dims, scaler in self.tensors.values():
            if "grid" in dims and grid_shard_slice is not None:
                grid_index = dims.index("grid")
                if scaler.shape[grid_index] >= grid_shard_slice.stop:
                    slices = [slice(None)] * len(dims)
                    slices[grid_index] = grid_shard_slice
                    scaler = scaler[tuple(slices)]

            reshaped_scaler = reshape_scaler(dims, scaler, layout)

            # subset_indices narrows the variable axis
            # we only apply it to scalers that actually span that axis
            # scalers broadcast over variables (size-1 last axis, e.g. node_weights or time_steps)
            # already align with the subset and should not be reindexed
            if subset_indices is not None and subset_indices != (...,) and reshaped_scaler.shape[-1] != 1:
                reshaped_scaler = reshaped_scaler[subset_indices]

            out *= reshaped_scaler

        return out

    def __repr__(self):
        return (
            f"ScalerTensor:\n - With tensors  : {list(self._tensors.keys())}\n"
            f" - In dimensions : {list(self.specified_dimensions.values())}"
        )

    def __contains__(self, dimension: int | tuple[int] | str) -> bool:
        """Check if a scaler name, or a dimension (axis number, `TensorDim` or tuple of them) is being scaled."""
        if isinstance(dimension, tuple):
            return dimension in self.specified_dimensions.values()
        if isinstance(dimension, int | TensorDim):
            return self.has_dim(dimension)
        return dimension in self._tensors

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        """Iterate over tensors."""
        return iter(self.tensors)

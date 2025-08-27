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

import torch
from torch import nn
from typing_extensions import Self

from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def grad_scaler(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scaler, prepend=False) to register this hook.

    Parameters
    ----------
    module : nn.Module
        Loss object (not used)
    grad_in : tuple[torch.Tensor, ...]
        Loss gradients
    grad_out : tuple[torch.Tensor, ...]
        Output gradients (not used)

    Returns
    -------
    tuple[torch.Tensor, ...]
        Re-scaled input gradients

    """
    del module, grad_out
    # first grad_input is that of the predicted state and the second is that of the "ground truth" (== zero)
    channels = grad_in[0].shape[-1]  # number of channels
    channel_weights = torch.reciprocal(torch.sum(torch.abs(grad_in[0]), dim=1, keepdim=True))  # channel-wise weights
    new_grad_in = (
        (channels * channel_weights) / torch.sum(channel_weights, dim=-1, keepdim=True) * grad_in[0]
    )  # rescaled gradient
    return new_grad_in, grad_in[1]


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

    When `get_scaler` or `scale` is called, the class will return the product of all scalers, resolved
    to the dimensional size of the input tensor.

    Additionally, the class can be subsetted to only return a subset of the scalers, but
    only from those given names.

    Examples
    --------
    >>> tensor = torch.randn(3, 4, 5)
    >>> scalers = ScaleTensor((0, torch.randn(3)), (1, torch.randn(4)))
    >>> scaled_tensor = scalers.scale(tensor)
    >>> scalers.get_scaler(tensor.ndim).shape
    torch.Size([3, 4, 1])
    >>> scalers.add_scaler(-1, torch.randn(5))
    >>> scalers.get_scaler(tensor.ndim).shape
    torch.Size([3, 4, 5])
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
        tensors = {}
        for name, (dimension, _) in self._tensors.items():
            tensors[name] = (dimension, self._buffers[name])

        return tensors

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

    def validate_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor) -> None:
        """Check if the scaler is compatible with the given dimension.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimensions to check `scaler` against
        scaler : torch.Tensor
            Scaler tensor to check

        Raises
        ------
        ValueError
            If the scaler is not compatible with the given dimension
        """
        if isinstance(dimension, int):
            dimension = [dimension]

        for scaler_dim, dim in enumerate(dimension):
            if dim not in self or scaler.shape[scaler_dim] == 1 or self.shape[dim] == 1 or dim == TensorDim.GRID:
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

        if isinstance(dimension, int):
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

        self._tensors[name] = (dimension, None)
        self.register_buffer(name, scaler, persistent=False)

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
            self._buffers.pop(scaler_to_pop, None)
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

        if name not in self._tensors:
            msg = f"scaler {name!r} not found in scalers."
            raise ValueError(msg)

        dimension = self._tensors[name][0]

        original_scaler = self._tensors.pop(name)
        original_scaler_buffer = self._buffers.pop(name, None)

        if not override:
            self.validate_scaler(dimension, scaler)

        try:
            self.add_scaler(dimension, scaler, name=name)
        except ValueError:
            self._tensors[name] = original_scaler
            self.register_buffer(name, original_scaler_buffer, persistent=False)
            raise

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
            Name/s or dimension/s of the scalers to get

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scaler_identifier, str | int):
            scaler_identifier = [scaler_identifier]
        if any(isinstance(scaler, int) for scaler in scaler_identifier):
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

    def subset_by_dim(self, dimensions: int | Sequence[int]) -> Self:
        """Get subset of the scalers, filtering by dimension.

        See `.subset` for subsetting by name.

        Parameters
        ----------
        dimensions : int | Sequence[int]
            Dimensions to get scalers of

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalers: dict[str, TENSOR_SPEC] = {}

        if isinstance(dimensions, int):
            dimensions = (dimensions,)

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
            Name/s or dimension/s of the scalers to exclude

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scaler_identifier, str | int):
            scaler_identifier = [scaler_identifier]
        if any(isinstance(scaler, int) for scaler in scaler_identifier):
            return self.without_by_dim(scaler_identifier)
        return self.without_by_str(scaler_identifier)

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

    def without_by_dim(self, dimensions: int | Sequence[int]) -> Self:
        """Get subset of the scalers, filtering out by dimension.

        Parameters
        ----------
        dimensions : int | Sequence[int]
            Dimensions to exclude scalers of

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalers: dict[str, TENSOR_SPEC] = {}

        if isinstance(dimensions, int):
            dimensions = (dimensions,)

        for name, (dim, scaler) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) == 0:
                subset_scalers[name] = (dim, scaler)

        return ScaleTensor(**subset_scalers)

    def resolve(self, ndim: int) -> Self:
        """Resolve relative indexes in scalers by associating against ndim.

        i.e. if a scaler was given as effecting dimension -1,
        and `ndim` was provided as 4, the scaler will be fixed
        to effect dimension 3.

        Parameters
        ----------
        ndim : int
            Number of dimensions to resolve relative indexing against

        Returns
        -------
        ScaleTensor
            ScaleTensor with all relative indexes resolved
        """
        resolved_scalers: dict[str, TENSOR_SPEC] = {}

        for name, (dims, scaler) in self.tensors.items():
            if any(d < 0 for d in dims):
                dims = [d if d >= 0 else ndim + d for d in dims]
            resolved_scalers[name] = (dims, scaler)

        return ScaleTensor(**resolved_scalers)

    def scale_iteratively(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...] | None = None,
        *,
        grid_shard_slice: slice | None = None,
    ) -> None:
        """Apply the scalers iteratively to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to scale
        subset_indices : tuple[int, ...] | None, optional
            Indices to subset the input tensor, by default None
        grid_shard_slice : slice | None, optional
            Slice to apply to the grid dimension, by default None
        """
        x_subset = x[subset_indices] if subset_indices is not None else x
        out = x_subset.clone()
        ndim = x.ndim
        tensors = self.resolve(ndim).tensors

        for dims, scaler in tensors.values():
            if TensorDim.GRID in dims and grid_shard_slice is not None:
                grid_index = dims.index(TensorDim.GRID)
                if scaler.shape[grid_index] >= grid_shard_slice.stop:
                    slices = [slice(None)] * len(dims)
                    slices[grid_index] = grid_shard_slice
                    scaler = scaler[tuple(slices)]

            missing_dims = [d for d in range(ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scaler.shape)

            reshaped_scaler = scaler.reshape(reshape)
            reshaped_scaler = torch.moveaxis(reshaped_scaler, list(range(ndim)), (*missing_dims, *dims))

            reshaped_scaler = reshaped_scaler.expand_as(x)

            if subset_indices is not None:
                reshaped_scaler = reshaped_scaler[subset_indices]

            out = out * reshaped_scaler

        return out

    def scale(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...] | None = None,
        *,
        grid_shard_slice: slice | None = None,
    ) -> None:
        """Scale a given tensor by the scalers.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to scale

        Returns
        -------
        torch.Tensor
            Scaled tensor
        """
        x_subset = x[subset_indices] if subset_indices is not None else x
        scaler = self.get_scaler(x_subset.ndim)
        if grid_shard_slice is not None and scaler.shape[TensorDim.GRID] > 1:
            slices = [slice(None)] * x_subset.ndim
            slices[TensorDim.GRID] = grid_shard_slice
            scaler = scaler[tuple(slices)]

        return x_subset * scaler

    def get_scaler(self, ndim: int, device: str | None = None) -> torch.Tensor:
        """Get completely resolved scaler tensor.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the tensor to resolve the scalers to
            Used to resolve relative indices, and add singleton dimensions
        device: str | None, optional
            Device to move the scaler to, by default None

        Returns
        -------
        torch.Tensor
            Scaler tensor

        Raises
        ------
        ValueError
            If resolving relative indices is invalid
        """
        complete_scaler = None

        tensors = self.resolve(ndim).tensors

        for dims, scaler in tensors.values():
            missing_dims = [d for d in range(ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scaler.shape)

            reshaped_scaler = scaler.reshape(reshape)
            reshaped_scaler = torch.moveaxis(reshaped_scaler, list(range(ndim)), (*missing_dims, *dims))

            complete_scaler = reshaped_scaler if complete_scaler is None else complete_scaler * reshaped_scaler

        complete_scaler = torch.ones(1) if complete_scaler is None else complete_scaler

        if device is not None:
            return complete_scaler.to(device)
        return complete_scaler

    def __mul__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.scale(tensor)

    def __rmul__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.scale(tensor)

    def __repr__(self):
        return (
            f"ScalerTensor:\n - With tensors  : {list(self._tensors.keys())}\n"
            f" - In dimensions : {list(self.specified_dimensions.values())}"
        )

    def __contains__(self, dimension: int | tuple[int] | str) -> bool:
        """Check if either scaler by name or dimension by int/tuple is being scaled."""
        if isinstance(dimension, tuple):
            return dimension in self.specified_dimensions.values()
        if isinstance(dimension, str):
            return dimension in self._tensors

        result = False
        for dim_assign, _ in self._tensors.values():
            result = dimension in dim_assign or result
        return result

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        """Iterate over tensors."""
        return iter(self.tensors)

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


def grad_scaler(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scalar, prepend=False) to register this hook.

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


TENSOR_SPEC = tuple[Union[int, tuple[int]], torch.Tensor]


class Shape:
    """Shape resolving object."""

    def __init__(self, func: Callable[[int], int]):
        self.func = func

    def __getitem__(self, dimension: int) -> int:
        return self.func(dimension)


# TODO(Harrison Cook): Consider moving this to subclass from a pytorch object and allow for device moving completely
class ScaleTensor:
    """Dynamically resolved tensor scaling class.

    Allows a user to specify a scalar and the dimensions it should be applied to.
    The class will then enforce that additional scalars are compatible with the specified dimensions.

    When `get_scalar` or `scale` is called, the class will return the product of all scalars, resolved
    to the dimensional size of the input tensor.

    Additionally, the class can be subsetted to only return a subset of the scalars, but
    only from those given names.

    Examples
    --------
    >>> tensor = torch.randn(3, 4, 5)
    >>> scalars = ScaleTensor((0, torch.randn(3)), (1, torch.randn(4)))
    >>> scaled_tensor = scalars.scale(tensor)
    >>> scalars.get_scalar(tensor.ndim).shape
    torch.Size([3, 4, 1])
    >>> scalars.add_scalar(-1, torch.randn(5))
    >>> scalars.get_scalar(tensor.ndim).shape
    torch.Size([3, 4, 5])
    """

    tensors: dict[str, TENSOR_SPEC]
    _specified_dimensions: dict[str, tuple[int]]

    def __init__(
        self,
        scalars: dict[str, TENSOR_SPEC] | TENSOR_SPEC | None = None,
        *tensors: TENSOR_SPEC,
        **named_tensors: dict[str, TENSOR_SPEC],
    ):
        """ScaleTensor constructor.

        Parameters
        ----------
        scalars : dict[str, TENSOR_SPEC] | TENSOR_SPEC | None, optional
            Scalars to initalise with, by default None
        tensors : TENSOR_SPEC
            Args form of (dimension, tensor) to add to the scalars
            Will be given a random uuid name
        named_tensors : dict[str, TENSOR_SPEC]
            Kwargs form of {name: (dimension, tensor)} to add to the scalars
        """
        self.tensors = {}
        self._specified_dimensions = {}

        named_tensors.update(scalars or {})
        self.add(named_tensors)

        for tensor_spec in tensors:
            self.add_scalar(*tensor_spec)

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

            unique_dims = {dim for dim_assign in self._specified_dimensions.values() for dim in dim_assign}
            error_msg = (
                f"Could not find shape of dimension {dimension}. "
                f"Tensors are only specified for dimensions {list(unique_dims)}."
            )
            raise IndexError(error_msg)

        return Shape(get_dim_shape)

    def validate_scalar(self, dimension: int | tuple[int], scalar: torch.Tensor) -> None:
        """Check if the scalar is compatible with the given dimension.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimensions to check `scalar` against
        scalar : torch.Tensor
            Scalar tensor to check

        Raises
        ------
        ValueError
            If the scalar is not compatible with the given dimension
        """
        if isinstance(dimension, int):
            dimension = [dimension]

        for scalar_dim, dim in enumerate(dimension):
            if dim not in self or scalar.shape[scalar_dim] == 1 or self.shape[dim] == 1:
                continue

            if self.shape[dim] != scalar.shape[scalar_dim]:
                error_msg = (
                    f"Incoming scalar shape {scalar.shape} at dimension {scalar_dim} "
                    f"does not match shape of saved scalar. Expected {self.shape[dim]}"
                )
                raise ValueError(error_msg)

    def add_scalar(
        self,
        dimension: int | tuple[int],
        scalar: torch.Tensor,
        *,
        name: str | None = None,
    ) -> ScaleTensor:
        """Add new scalar to be applied along `dimension`.

        Dimension can be a single int even for a multi-dimensional scalar,
        in this case the dimensions are assigned as a range starting from the given int.
        Negative indexes are also valid, and will be resolved against the tensor's ndim.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimension/s to apply the scalar to
        scalar : torch.Tensor
            Scalar tensor to apply
        name : str | None, optional
            Name of the scalar, by default None

        Returns
        -------
        ScaleTensor
            ScaleTensor with the scalar removed
        """
        if not isinstance(scalar, torch.Tensor):
            scalar = torch.tensor([scalar]) if isinstance(scalar, (int, float)) else torch.tensor(scalar)

        if isinstance(dimension, int):
            if len(scalar.shape) == 1:
                dimension = (dimension,)
            else:
                dimension = tuple(dimension + i for i in range(len(scalar.shape)))
        else:
            dimension = tuple(dimension)

        if name is None:
            name = str(uuid.uuid4())

        if name in self.tensors:
            msg = f"Scalar {name!r} already exists in scalars."
            raise ValueError(msg)

        try:
            self.validate_scalar(dimension, scalar)
        except ValueError as e:
            error_msg = f"Validating tensor {name!r} raised an error."
            raise ValueError(error_msg) from e

        self.tensors[name] = (dimension, scalar)
        self._specified_dimensions[name] = dimension

        return self

    def remove_scalar(self, scalar_to_remove: str | int) -> ScaleTensor:
        """
        Remove scalar from ScaleTensor.

        Parameters
        ----------
        scalar_to_remove : str | int
            Name or index of tensor to remove

        Raises
        ------
        ValueError
            If the scalar is not in the scalars

        Returns
        -------
        ScaleTensor
            ScaleTensor with the scalar removed
        """
        for scalar_to_pop in self.subset(scalar_to_remove).tensors:
            self.tensors.pop(scalar_to_pop)
            self._specified_dimensions.pop(scalar_to_pop)
        return self

    def freeze_state(self) -> FrozenStateRecord:  # noqa: F821
        """
        Freeze the state of the Scalar with a context manager.

        Any changes made will be reverted on exit.

        Returns
        -------
        FrozenStateRecord
            Context manager to freeze the state of this ScaleTensor
        """
        record_of_scalars: dict = self.tensors.copy()

        class FrozenStateRecord:
            """Freeze the state of the ScaleTensor. Any changes will be reverted on exit."""

            def __enter__(self):
                pass

            def __exit__(context_self, *a):  # noqa: N805
                for key in list(self.tensors.keys()):
                    if key not in record_of_scalars:
                        self.remove_scalar(key)

                for key in record_of_scalars:
                    if key not in self:
                        self.add_scalar(*record_of_scalars[key], name=key)

        return FrozenStateRecord()

    def update_scalar(self, name: str, scalar: torch.Tensor, *, override: bool = False) -> None:
        """Update an existing scalar maintaining original dimensions.

        If `override` is False, the scalar must be valid against the original dimensions.
        If `override` is True, the scalar will be updated regardless of validity against original scalar.

        Parameters
        ----------
        name : str
            Name of the scalar to update
        scalar : torch.Tensor
            New scalar tensor
        override : bool, optional
            Whether to override the scalar ignoring dimension compatibility, by default False
        """
        if name not in self.tensors:
            msg = f"Scalar {name!r} not found in scalars."
            raise ValueError(msg)

        dimension = self.tensors[name][0]

        if not override:
            self.validate_scalar(dimension, scalar)

        original_scalar = self.tensors.pop(name)
        original_dimension = self._specified_dimensions.pop(name)

        try:
            self.add_scalar(dimension, scalar, name=name)
        except ValueError:
            self.tensors[name] = original_scalar
            self._specified_dimensions[name] = original_dimension
            raise

    def add(self, new_scalars: dict[str, TENSOR_SPEC] | list[TENSOR_SPEC] | None = None, **kwargs) -> None:
        """Add multiple scalars to the existing scalars.

        Parameters
        ----------
        new_scalars : dict[str, TENSOR_SPEC] | list[TENSOR_SPEC] | None, optional
            Scalars to add, see `add_scalar` for more info, by default None
        **kwargs:
            Kwargs form of {name: (dimension, tensor)} to add to the scalars
        """
        if isinstance(new_scalars, list):
            for tensor_spec in new_scalars:
                self.add_scalar(*tensor_spec)
        else:
            kwargs.update(new_scalars or {})
        for name, tensor_spec in kwargs.items():
            self.add_scalar(*tensor_spec, name=name)

    def update(self, updated_scalars: dict[str, torch.Tensor] | None = None, override: bool = False, **kwargs) -> None:
        """Update multiple scalars in the existing scalars.

        If `override` is False, the scalar must be valid against the original dimensions.
        If `override` is True, the scalar will be updated regardless of shape.

        Parameters
        ----------
        updated_scalars : dict[str, torch.Tensor] | None, optional
            Scalars to update, referenced by name, by default None
        override : bool, optional
            Whether to override the scalar ignoring dimension compatibility, by default False
        **kwargs:
            Kwargs form of {name: tensor} to update in the scalars
        """
        kwargs.update(updated_scalars or {})
        for name, tensor in kwargs.items():
            self.update_scalar(name, tensor, override=override)

    def subset(self, scalar_identifier: str | Sequence[str] | int | Sequence[int]) -> ScaleTensor:
        """Get subset of the scalars, filtering by name or dimension.

        Parameters
        ----------
        scalar_identifier : str | Sequence[str] | int | Sequence[int]
            Name/s or dimension/s of the scalars to get

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalar_identifier, (str, int)):
            scalar_identifier = [scalar_identifier]
        if any(isinstance(scalar, int) for scalar in scalar_identifier):
            return self.subset_by_dim(scalar_identifier)
        return self.subset_by_str(scalar_identifier)

    def subset_by_str(self, scalars: str | Sequence[str]) -> ScaleTensor:
        """Get subset of the scalars, filtering by name.

        See `.subset_by_dim` for subsetting by affected dimensions.

        Parameters
        ----------
        scalars : str | Sequence[str]
            Name/s of the scalars to get

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalars, str):
            scalars = [scalars]
        return ScaleTensor(**{name: self.tensors[name] for name in scalars})

    def subset_by_dim(self, dimensions: int | Sequence[int]) -> ScaleTensor:
        """Get subset of the scalars, filtering by dimension.

        See `.subset` for subsetting by name.

        Parameters
        ----------
        dimensions : int | Sequence[int]
            Dimensions to get scalars of

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalars: dict[str, TENSOR_SPEC] = {}

        if isinstance(dimensions, int):
            dimensions = (dimensions,)

        for name, (dim, scalar) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) > 0:
                subset_scalars[name] = (dim, scalar)

        return ScaleTensor(**subset_scalars)

    def without(self, scalar_identifier: str | Sequence[str] | int | Sequence[int]) -> ScaleTensor:
        """Get subset of the scalars, filtering out by name or dimension.

        Parameters
        ----------
        scalar_identifier : str | Sequence[str] | int | Sequence[int]
            Name/s or dimension/s of the scalars to exclude

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalar_identifier, (str, int)):
            scalar_identifier = [scalar_identifier]
        if any(isinstance(scalar, int) for scalar in scalar_identifier):
            return self.without_by_dim(scalar_identifier)
        return self.without_by_str(scalar_identifier)

    def without_by_str(self, scalars: str | Sequence[str]) -> ScaleTensor:
        """Get subset of the scalars, filtering out by name.

        Parameters
        ----------
        scalars : str | Sequence[str]
            Name/s of the scalars to exclude

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalars, str):
            scalars = [scalars]
        return ScaleTensor(**{name: tensor for name, tensor in self.tensors.items() if name not in scalars})

    def without_by_dim(self, dimensions: int | Sequence[int]) -> ScaleTensor:
        """Get subset of the scalars, filtering out by dimension.

        Parameters
        ----------
        dimensions : int | Sequence[int]
            Dimensions to exclude scalars of

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalars: dict[str, TENSOR_SPEC] = {}

        if isinstance(dimensions, int):
            dimensions = (dimensions,)

        for name, (dim, scalar) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) == 0:
                subset_scalars[name] = (dim, scalar)

        return ScaleTensor(**subset_scalars)

    def resolve(self, ndim: int) -> ScaleTensor:
        """Resolve relative indexes in scalars by associating against ndim.

        i.e. if a scalar was given as effecting dimension -1,
        and `ndim` was provided as 4, the scalar will be fixed
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
        resolved_scalars: dict[str, TENSOR_SPEC] = {}

        for name, (dims, scalar) in self.tensors.items():
            if any(d < 0 for d in dims):
                dims = [d if d >= 0 else ndim + d for d in dims]
            resolved_scalars[name] = (dims, scalar)

        return ScaleTensor(**resolved_scalars)

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale a given tensor by the scalars.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to scale

        Returns
        -------
        torch.Tensor
            Scaled tensor
        """
        return tensor * self.get_scalar(tensor.ndim, device=tensor.device)

    def get_scalar(self, ndim: int, device: str | None = None) -> torch.Tensor:
        """Get completely resolved scalar tensor.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the tensor to resolve the scalars to
            Used to resolve relative indices, and add singleton dimensions
        device: str | None, optional
            Device to move the scalar to, by default None

        Returns
        -------
        torch.Tensor
            Scalar tensor

        Raises
        ------
        ValueError
            If resolving relative indices is invalid
        """
        complete_scalar = None

        tensors = self.resolve(ndim).tensors

        for dims, scalar in tensors.values():
            missing_dims = [d for d in range(ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scalar.shape)

            reshaped_scalar = scalar.reshape(reshape)
            reshaped_scalar = torch.moveaxis(reshaped_scalar, list(range(ndim)), (*missing_dims, *dims))

            complete_scalar = reshaped_scalar if complete_scalar is None else complete_scalar * reshaped_scalar

        complete_scalar = torch.ones(1) if complete_scalar is None else complete_scalar

        if device is not None:
            return complete_scalar.to(device)
        return complete_scalar

    def to(self, *args, **kwargs) -> None:
        """Move scalars inplace."""
        for name, (dims, tensor) in self.tensors.items():
            self.tensors[name] = (dims, tensor.to(*args, **kwargs))

    def __mul__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.scale(tensor)

    def __rmul__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.scale(tensor)

    def __repr__(self):
        return (
            f"ScalarTensor:\n - With tensors  : {list(self.tensors.keys())}\n"
            f" - In dimensions : {list(self._specified_dimensions.values())}"
        )

    def __contains__(self, dimension: int | tuple[int] | str) -> bool:
        """Check if either scalar by name or dimension by int/tuple is being scaled."""
        if isinstance(dimension, tuple):
            return dimension in self._specified_dimensions.values()
        if isinstance(dimension, str):
            return dimension in self.tensors

        result = False
        for dim_assign, _ in self.tensors.values():
            result = dimension in dim_assign or result
        return result

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        """Iterate over tensors."""
        return iter(self.tensors)

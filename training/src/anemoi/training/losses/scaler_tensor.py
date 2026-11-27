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
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Self

import torch
from torch import distributed as dist
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Source
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


LOSS_SCALE_REL_EPS = 1e-8
"""Variables whose gradient norm is below this fraction of the largest variable norm are treated as inactive."""


class LossScaleFunction(Function):
    """Identity in the forward pass; equalises the per-variable loss gradients in the backward pass.

    Implements the loss-gradient rescaling of https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2. The
    function takes one or more tensors ("segments") of equal rank and variable width. For every segment
    the L2 norm of the gradient of each variable is taken over ``reduce_dims``; the remaining non-variable
    dimensions index independent rows (e.g. (batch, time) for a gridded source, or one row per sample of a
    tabular source). Within each row, the gradient of each variable is rescaled so that all active variables
    end up with the same norm.

    Variables whose gradient is (numerically) zero, e.g. outputs not seen by the loss or fully masked
    observations, are left at zero and do not take part in the normalisation. The rescaling is
    positively homogeneous of degree one, so it does not depend on the absolute scale of the loss.

    All segments are handled by a single autograd node, so the norms need at most one all-reduce over the
    model-parallel group. That all-reduce is required when the reduced dimensions are sharded, and harmless
    when they are replicated: it multiplies every norm of a row by the same factor, and the weights only
    depend on relative norms.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        reduce_dims: tuple[int, ...],
        var_dim: int,
        power: float,
        max_gain: float | None,
        group: ProcessGroup | None,
        *xs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return the inputs unchanged and store the backward options.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        reduce_dims : tuple[int, ...]
            Dimensions the per-variable gradient norm is taken over (non-negative).
        var_dim : int
            Variable dimension (non-negative).
        power : float
            Exponent of the inverse gradient norm; 1 equalises the norms, 0 is the identity.
        max_gain : float | None
            Upper bound on the weight of the weakest variable relative to the strongest one.
        group : ProcessGroup | None
            Model-parallel group to all-reduce the norms over when the reduced dimensions are sharded.
        *xs : torch.Tensor
            Segments, all of the same rank and variable width.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Views of the inputs, with this function's backward attached.
        """
        if not xs:
            msg = "LossScaleFunction needs at least one tensor."
            raise ValueError(msg)
        ndim = xs[0].ndim
        n_var = xs[0].shape[var_dim]
        if any(x.ndim != ndim or x.shape[var_dim] != n_var for x in xs):
            msg = "LossScaleFunction needs tensors of equal rank and variable width."
            raise ValueError(msg)
        if not 0 <= var_dim < ndim or any(not 0 <= d < ndim for d in reduce_dims):
            msg = f"Dimensions out of range for {ndim}D tensors: reduce_dims={reduce_dims}, var_dim={var_dim}."
            raise ValueError(msg)
        if var_dim in reduce_dims:
            msg = f"The variable dimension {var_dim} cannot be reduced over."
            raise ValueError(msg)
        if power < 0:
            msg = f"power must be non-negative, got {power}."
            raise ValueError(msg)

        ctx.reduce_dims = tuple(reduce_dims)
        ctx.var_dim = var_dim
        ctx.power = power
        ctx.max_gain = max_gain
        ctx.group = group
        return tuple(x.view_as(x) for x in xs)

    @staticmethod
    @once_differentiable
    def backward(ctx: torch.autograd.function.FunctionCtx, *grads: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Rescale the incoming gradients per variable.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        *grads : torch.Tensor
            Gradients of the loss with respect to each segment.

        Returns
        -------
        tuple[torch.Tensor | None, ...]
            None for the five non-tensor arguments, then the rescaled gradient of each segment.
        """
        var_dim = ctx.var_dim
        # Low-precision gradients are upcast to float32; float64 is kept.
        gs = [grad.to(torch.promote_types(grad.dtype, torch.float32)) for grad in grads]
        n_var = gs[0].shape[var_dim]

        # Squared per-variable norms of every row, flattened to (rows, variables) and stacked over segments.
        row_shapes, flat_sq_norms = [], []
        for g in gs:
            sq_norm = g.square()
            if ctx.reduce_dims:
                sq_norm = sq_norm.sum(dim=ctx.reduce_dims, keepdim=True)
            sq_norm = sq_norm.movedim(var_dim, -1)
            row_shapes.append(sq_norm.shape)
            flat_sq_norms.append(sq_norm.reshape(-1, n_var))
        sq_norm = torch.cat(flat_sq_norms)

        if ctx.group is not None and dist.get_world_size(group=ctx.group) > 1:
            dist.all_reduce(sq_norm, group=ctx.group)
        norm = sq_norm.sqrt()

        tiny = torch.finfo(norm.dtype).tiny
        rel_norm = norm / norm.amax(dim=-1, keepdim=True).clamp_min(tiny)
        active = rel_norm > LOSS_SCALE_REL_EPS
        weight = torch.where(active, rel_norm.clamp_min(LOSS_SCALE_REL_EPS).pow(-ctx.power), 0.0)
        if ctx.max_gain is not None:
            weight = weight.clamp_max(ctx.max_gain)

        n_active = active.sum(dim=-1, keepdim=True)
        weight = n_active * weight / weight.sum(dim=-1, keepdim=True).clamp_min(tiny)

        rows = [flat.shape[0] for flat in flat_sq_norms]
        scaled = [
            (g * w.reshape(shape).movedim(-1, var_dim)).to(grad.dtype)
            for g, grad, w, shape in zip(gs, grads, weight.split(rows), row_shapes, strict=True)
        ]
        return (None, None, None, None, None, *scaled)


class LossScaler(nn.Module):
    """Module wrapper around `LossScaleFunction`.

    Works on a gridded or tabular `Source`, whose layout gives the axes: the per-variable gradient norm is
    taken over the ensemble and grid axes (whichever the layout has), separately for every index of the
    remaining axes. For gridded data that is one norm per (batch, time) row; for tabular data, where each
    sample is its own tensor and time is folded into the grid axis, one norm per sample.

    The module holds no parameters or buffers, so enabling or disabling it does not change the state dict.
    """

    def __init__(self, power: float = 1.0, max_gain: float | None = None) -> None:
        """Initialise LossScaler.

        Parameters
        ----------
        power : float, optional
            Exponent of the inverse gradient norm; 1 equalises the norms, 0 is the identity. By default 1.0.
        max_gain : float | None, optional
            Upper bound on the weight of the weakest variable relative to the strongest one, by default None.
        """
        super().__init__()
        if power < 0:
            msg = f"power must be non-negative, got {power}."
            raise ValueError(msg)
        if max_gain is not None and max_gain < 1:
            msg = f"max_gain must be at least 1, got {max_gain}."
            raise ValueError(msg)
        self.power = power
        self.max_gain = max_gain

    def _scale(
        self,
        tensors: Sequence[torch.Tensor],
        layout: TensorLayout,
        group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, ...]:
        ndim = tensors[0].ndim
        var_dim = layout.axis(TensorDim.VARIABLE, ndim=ndim)
        reduce_dims = tuple(
            layout.axis(name, ndim=ndim) for name in (TensorDim.ENSEMBLE_DIM, TensorDim.GRID) if layout.has_axis(name)
        )
        out = LossScaleFunction.apply(reduce_dims, var_dim, self.power, self.max_gain, group, *tensors)
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)

    def forward(self, source: Source, group: ProcessGroup | None = None) -> Source:
        """Apply the gradient rescaling to the predictions.

        Parameters
        ----------
        source : Source
            Predictions, gridded or tabular. The reduced and variable axes are read from its layout.
        group : ProcessGroup | None, optional
            Model-parallel group, passed only when the source is sharded. By default None.

        Returns
        -------
        Source
            The same source, with rescaled gradients in the backward pass.
        """
        if not isinstance(source, Source):
            msg = (
                f"LossScaler expects a Source (gridded or tabular), got {type(source).__name__}: "
                "the axes to reduce over are taken from the source layout."
            )
            raise TypeError(msg)
        if source.is_tabular:
            if not source.values:
                return source
            return source.clone(values=list(self._scale(source.values, source.layout, group)))
        return source.clone(values=self._scale([source.values], source.layout, group)[0])

    @classmethod
    def from_config(cls, config: object) -> "LossScaler | None":
        """Build a LossScaler from `training.loss_gradient_scaling`, or None if disabled.

        Parameters
        ----------
        config : object
            A bool (legacy form), or a mapping or object with `enabled`, and optionally `power` and `max_gain`.

        Returns
        -------
        LossScaler | None
            The scaler, or None if gradient scaling is disabled.
        """
        if config is None or isinstance(config, bool):
            return cls() if config else None

        def get(key: str, default: Any) -> Any:
            if isinstance(config, Mapping):
                return config.get(key, default)
            return getattr(config, key, default)

        if not get("enabled", False):
            return None
        return cls(power=get("power", 1.0), max_gain=get("max_gain", None))

    def extra_repr(self) -> str:
        return f"power={self.power}, max_gain={self.max_gain}"


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
        assert not scaler.requires_grad, (
            f"Scaler tensors must not require gradients. Got requires_grad=True for scaler {name!r}."
        )

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
        assert not scaler.requires_grad, (
            f"Scaler tensors must not require gradients. Got requires_grad=True for scaler {name!r}."
        )

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

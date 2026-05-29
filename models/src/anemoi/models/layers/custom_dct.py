# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import math
from typing import Callable
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_DCT2D_BACKENDS: dict[str, type[nn.Module]] = {}


def register_dct2d_backend(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Class decorator: register a DCT2D backend factory under *name*.

    Example
    -------
    ::

        @register_dct2d_backend("my_backend")
        class MyBackend(nn.Module):
            def __init__(self, y_dim, x_dim, **kwargs): ...
            def forward(self, x: Tensor) -> Tensor: ...  # (B, y_dim, x_dim) -> (B, y_dim, x_dim)
    """

    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _DCT2D_BACKENDS:
            raise ValueError(f"DCT2D backend {name!r} is already registered.")
        _DCT2D_BACKENDS[name] = cls
        return cls

    return deco


def build_dct2d_backend(name: str, **kwargs: object) -> nn.Module:
    """Instantiate a registered DCT2D backend by *name*.

    Parameters
    ----------
    name:
        Key used when registering the backend with
        :func:`register_dct2d_backend`.
    **kwargs:
        Forwarded to the backend constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in _DCT2D_BACKENDS:
        raise ValueError(f"Unknown DCT2D backend {name!r}. " f"Available: {sorted(_DCT2D_BACKENDS)}")
    return _DCT2D_BACKENDS[name](**kwargs)


# ---------------------------------------------------------------------------
# Built-in backend: torch_dct (default, unchanged behaviour)
# ---------------------------------------------------------------------------


@register_dct2d_backend("torch_dct")
class _TorchDCTBackend(nn.Module):
    """Wrapper around the third-party ``torch_dct.dct_2d``."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        try:
            from torch_dct import dct_2d
        except ImportError as err:
            raise ImportError(
                "torch_dct is required for the 'torch_dct' backend. " "Install it or switch to backend='custom_dct'."
            ) from err
        return dct_2d(x)


# ---------------------------------------------------------------------------
# Built-in backend: custom_dct
# ---------------------------------------------------------------------------


@register_dct2d_backend("custom_dct")
class DCT2DCore(nn.Module):
    """2D DCT-II via cached basis matrices.

    Computes ``Y = D_y @ X @ D_x^T`` using two batched GEMMs where
    ``D_y`` (shape ``y_dim × y_dim``) and ``D_x`` (shape ``x_dim × x_dim``)
    are the DCT-II basis matrices stored as non-persistent buffers.

    Parameters
    ----------
    y_dim:
        Height of the 2D spatial field.
    x_dim:
        Width of the 2D spatial field.
    norm:
        ``"unnormalized"`` (default) — matches ``torch_dct`` / scipy.
        ``"ortho"`` — orthonormal; ``D^T @ D = I``
    compute_dtype:
        If set, cast inputs to this dtype before the matmul and cast
        the output back.  Useful for preserving numerical accuracy when
        the model runs in bf16 / fp16 (e.g. ``torch.float32``).
        ``None`` means use the input dtype throughout.
    """

    def __init__(
        self,
        y_dim: int,
        x_dim: int,
        *,
        norm: Literal["unnormalized", "ortho"] = "unnormalized",
        compute_dtype: torch.dtype | None = None,
        **_kwargs: object,
    ) -> None:
        super().__init__()

        del _kwargs

        if norm not in ("unnormalized", "ortho"):
            raise ValueError(f"norm must be 'unnormalized' or 'ortho', got {norm!r}")

        self.y_dim = y_dim
        self.x_dim = x_dim
        self.norm = norm
        self.compute_dtype = compute_dtype

        # Non-persistent: reproducible from (y_dim, x_dim, norm) so we
        # don't bloat checkpoints.
        self.register_buffer("D_y", self._build_dct_matrix(y_dim, norm), persistent=False)
        self.register_buffer("D_x", self._build_dct_matrix(x_dim, norm), persistent=False)

        LOGGER.debug(
            "DCT2DCore: y_dim=%d x_dim=%d norm=%s compute_dtype=%s",
            y_dim,
            x_dim,
            norm,
            compute_dtype,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dct_matrix(n: int, norm: Literal["unnormalized", "ortho"]) -> Tensor:
        """Build the n×n DCT-II basis matrix.

        Parameters
        ----------
        n:
            Transform length.
        norm:
            ``"ortho"`` — orthonormal; ``D^T @ D = I``.
            ``"unnormalized"`` — matches ``torch_dct`` / scipy default.

        Returns
        -------
        Tensor
            Shape ``(n, n)``, dtype ``torch.float32``.
        """
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)  # (n, 1) — freq index
        i = torch.arange(n, dtype=torch.float64).unsqueeze(0)  # (1, n) — space index
        D = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))  # (n, n)

        if norm == "ortho":
            D[0].mul_(1.0 / math.sqrt(n))
            D[1:].mul_(math.sqrt(2.0 / n))
        else:
            D.mul_(2.0)

        return D.to(torch.float32)

    def _upcast(self, x: Tensor) -> tuple[Tensor, torch.dtype]:
        """Optionally upcast *x* to ``self.compute_dtype``."""
        src = x.dtype
        if self.compute_dtype is not None and src != self.compute_dtype:
            x = x.to(self.compute_dtype)
        return x, src

    def _get_matrices(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return ``(D_y, D_x)`` cast to *dtype*."""
        return self.D_y.to(dtype), self.D_x.to(dtype)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D DCT-II: ``Y = D_y @ X @ D_x^T``.

        Parameters
        ----------
        x:
            Shape ``(B, y_dim, x_dim)``.  Must be contiguous or will be made so.

        Returns
        -------
        Tensor
            Same shape ``(B, y_dim, x_dim)``.
        """
        x = x.contiguous()
        x, src_dtype = self._upcast(x)
        D_y, D_x = self._get_matrices(x.dtype)

        # Left-multiply along y:  (y_dim, y_dim) @ (B, y_dim, x_dim) -> (B, y_dim, x_dim)
        y = torch.matmul(D_y, x)
        # Right-multiply along x: (B, y_dim, x_dim) @ (x_dim, x_dim) -> (B, y_dim, x_dim)
        y = torch.matmul(y, D_x.t())

        if self.compute_dtype is not None:
            y = y.to(src_dtype)
        return y

    def inverse(self, x: Tensor) -> Tensor:
        """Apply 2D DCT-III (inverse of DCT-II for ``norm='ortho'``).

        Parameters
        ----------
        x:
            Shape ``(B, y_dim, x_dim)``.

        Returns
        -------
        Tensor
            Same shape ``(B, y_dim, x_dim)``.

        Raises
        ------
        NotImplementedError
            If ``self.norm == 'unnormalized'``.  Use ``torch_dct.idct_2d``
            or reconstruct with ``norm='ortho'`` instead.
        """
        if self.norm != "ortho":
            raise NotImplementedError(
                "DCT2DCore.inverse() is only implemented for norm='ortho'. "
                "For the unnormalized convention the inverse requires a "
                "non-trivial per-coefficient scaling; use "
                "torch_dct.idct_2d() or switch to norm='ortho'."
            )

        x = x.contiguous()
        x, src_dtype = self._upcast(x)
        D_y, D_x = self._get_matrices(x.dtype)

        # Left-multiply along y:  (y_dim, y_dim)^T @ (B, y_dim, x_dim) -> (B, y_dim, x_dim)
        y = torch.matmul(D_y.t(), x)
        # Right-multiply along x: (B, y_dim, x_dim) @ (x_dim, x_dim) -> (B, y_dim, x_dim)
        y = torch.matmul(y, D_x)

        if self.compute_dtype is not None:
            y = y.to(src_dtype)
        return y

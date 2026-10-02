# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

r"""Spherical random fields for ensemble perturbation, ported from FourCastNet 3.

This is a faithful port of ``makani/models/noise.py`` (NVIDIA, Apache-2.0) onto
anemoi's native spherical harmonic transforms, so that the generated fields live
directly on the flat node list of an anemoi graph rather than on a dense
``(nlat, nlon)`` image.

Three differences from the original are unavoidable and are called out at their
definitions:

1. The inverse transform is ``anemoi.models.layers.spectral_transforms`` rather
   than ``torch_harmonics``. Those use different Legendre normalisations, so
   ``sigma_l`` is rescaled by a calibration factor measured from the transform
   itself (see :func:`orthonormal_basis_scale`).
2. The state carries an explicit ensemble dimension, because anemoi holds all
   per-device ensemble members in one tensor instead of one member per rank.
3. Spatial model parallelism is handled by generating the full field and letting
   the caller shard it, rather than by a spectrally distributed transform.
"""

import logging
import math
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import amp
from torch import nn

from anemoi.models.layers.spectral_transforms import InverseOctahedralSHT
from anemoi.models.layers.spectral_transforms import InverseReducedSHT
from anemoi.models.layers.spectral_transforms import InverseRegularSHT
from anemoi.models.layers.spectral_transforms import InverseSpectralTransform

LOGGER = logging.getLogger(__name__)


def build_inverse_sht(
    grid: Union[str, int],
    lmax: Optional[int] = None,
    use_graphed_irfft: bool = False,
) -> tuple[InverseSpectralTransform, int, int]:
    r"""Build the inverse spherical harmonic transform for a named anemoi grid.

    Parameters
    ----------
    grid : str or int
        ``"n320"`` for a reduced Gaussian grid, ``"o96"`` for an octahedral
        reduced grid, or an integer ``nlat`` for a regular lon-lat grid.
    lmax : int, optional
        Number of spherical harmonic degrees to retain, i.e. ``l = 0 .. lmax-1``.
        Defaults to ``nlat``, matching the torch-harmonics default used by
        FourCastNet 3. Note the off-by-one against anemoi's ``truncation``
        argument, which sizes the Legendre array as ``truncation + 1``.
    use_graphed_irfft : bool, optional
        Use CUDA graphs for the per-ring inverse FFTs.

    Returns
    -------
    isht : InverseSpectralTransform
        The transform, mapping complex ``[..., lmax, lmax]`` to real ``[..., points]``.
    lmax : int
        The resolved number of degrees.
    num_grid_points : int
        Number of points the transform emits.
    """
    if isinstance(grid, int) or (isinstance(grid, str) and grid.isdigit()):
        nlat = int(grid)
        lmax = lmax or nlat
        isht = InverseRegularSHT(nlat=nlat, truncation=lmax - 1)
    elif grid[0] in ("n", "N"):
        nlat = 2 * int(grid[1:])
        lmax = lmax or nlat
        isht = InverseReducedSHT(grid=grid, truncation=lmax - 1, use_graphed_irfft=use_graphed_irfft)
    elif grid[0] in ("o", "O"):
        nlat = 2 * int(grid[1:])
        lmax = lmax or nlat
        isht = InverseOctahedralSHT(nlat=nlat, truncation=lmax - 1, use_graphed_irfft=use_graphed_irfft)
    else:
        raise ValueError(f"Unsupported grid specification '{grid}'. Expected 'nNNN', 'oNNN' or an integer nlat.")

    # legpoly builds this in float64; for lmax=640 that is a 2.1 GB buffer and the
    # per-call `.to(x.dtype)` in the transform would copy it every forward.
    isht._isht.pct = isht._isht.pct.to(torch.float32)

    return isht, lmax, isht._isht.n_grid_points


def orthonormal_basis_scale(isht: InverseSpectralTransform) -> float:
    r"""Factor converting orthonormal-basis coefficients to anemoi's basis.

    FourCastNet 3's ``sigma_l`` formulae are written against torch-harmonics'
    orthonormal spherical harmonics, where a unit :math:`a_{00}` coefficient
    produces a constant field of :math:`1/\sqrt{4\pi}`. Anemoi's
    ``spectral_helpers.legpoly`` uses the ECMWF :math:`4\pi` normalisation
    instead, so the same coefficient produces a different constant.

    Rather than hard-coding a factor that would silently rot if either
    convention changed, measure it: transform a unit :math:`a_{00}` and compare.
    The two bases differ by a single scalar (``legpoly`` applies its
    ``norm_factor`` globally), so one probe is sufficient -- and
    ``test_pointwise_variance_matches_sigma`` fails loudly if that ever stops
    being true.
    """
    lmax = isht._isht.truncation + 1
    device = isht._isht.pct.device
    coeffs = torch.zeros(1, lmax, lmax, dtype=torch.complex64, device=device)
    coeffs[0, 0, 0] = 1.0
    with torch.no_grad():
        constant_field = isht(coeffs)[0, 0].item()
    return (1.0 / math.sqrt(4.0 * math.pi)) / constant_field


def toep(c, r=None):
    r"""Construct a Toeplitz matrix from its first column and row.

    Every diagonal of the result is constant: ``T[i, j]`` depends only on
    ``i - j``. Used to build the discount matrix of powers of :math:`\phi` that
    correlates a freshly drawn noise history, so that resampling the whole
    history reproduces the temporal correlation of the process instead of giving
    independent steps.

    Vendored from SciPy to avoid a runtime dependency on it:
    https://github.com/scipy/scipy/blob/v1.13.0/scipy/linalg/_special_matrices.py#L17-L77
    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]

    return np.lib.stride_tricks.as_strided(vals[len(c) - 1 :], shape=out_shp, strides=(-n, n)).copy()


class BaseSphericalNoise(nn.Module):
    r"""Abstract base class for random fields on the sphere :math:`S^2`.

    Noise for an ensemble weather model has to be spatially correlated: white
    noise on a grid of nodes is neither isotropic (cells shrink toward the poles)
    nor physically plausible as a perturbation. All subclasses therefore generate
    noise in *spectral* space and transform it to the grid with an inverse SHT,
    which makes the correlation structure an explicit choice of angular power
    spectrum and keeps the field isotropic by construction.

    This base class owns the machinery every variant needs: the inverse SHT,
    one private RNG per ensemble member, and a ``state`` buffer holding the
    current spectral coefficients. Subclasses define the spectrum and decide
    whether they are stateful (the field evolves from step to step, as in
    :class:`DiffusionNoiseS2`) or stateless (each draw is independent, as in
    :class:`IsotropicGaussianRandomFieldS2`).

    The state buffer is non-persistent: it is a per-run scratch tensor and is
    deliberately kept out of checkpoints, so restoring one does not resurrect a
    stale noise realization.

    Parameters
    ----------
    isht : InverseSpectralTransform
        Transform mapping complex ``[..., lmax, lmax]`` to real ``[..., points]``.
    lmax : int
        Number of spherical harmonic degrees, ``l = 0 .. lmax-1``.
    num_grid_points : int
        Number of nodes the transform emits.
    batch_size : int
        Initial batch size of the state buffer. Resized on demand.
    ensemble_size : int
        Number of ensemble members held on this device.
    num_channels : int
        Number of noise channels.
    num_time_steps : int
        Number of time steps held in the state. Mirrors makani's ``n_history + 1``,
        i.e. anemoi's ``multistep_input``.
    seeds : list of int
        One seed per ensemble member. Members sharing a seed and differing in
        ``reflects`` form an antithetic pair.
    reflects : list of bool
        Per-member sign flip. Two members sharing a seed but differing in this
        flag produce exactly opposite perturbations, which cancels first-order
        sampling error in the ensemble mean.
    """

    def __init__(
        self,
        *,
        isht: InverseSpectralTransform,
        lmax: int,
        num_grid_points: int,
        batch_size: int,
        ensemble_size: int,
        num_channels: int,
        num_time_steps: int,
        seeds: list[int],
        reflects: list[bool],
        **kwargs,
    ) -> None:
        super().__init__()

        if len(seeds) != ensemble_size or len(reflects) != ensemble_size:
            raise ValueError(
                f"seeds ({len(seeds)}) and reflects ({len(reflects)}) must both have "
                f"ensemble_size ({ensemble_size}) entries."
            )

        self.isht = isht
        self.lmax = lmax
        self.mmax = lmax
        self.num_grid_points = num_grid_points
        self.ensemble_size = ensemble_size
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps
        self.seeds = list(seeds)
        self.reflects = list(reflects)

        # Converts the FourCastNet 3 sigma_l formulae into anemoi's basis.
        self.basis_scale = orthonormal_basis_scale(isht)

        self.set_rng(self.seeds)

        self._ensure_state(batch_size, device=torch.device("cpu"), dtype=torch.float32)

    @property
    def _state_shape_suffix(self) -> tuple[int, ...]:
        """Shape of the state buffer beyond the batch dim.

        Subclasses override this to customize the layout (e.g. :class:`DummyNoiseS2`
        stores state in spatial, not spectral, form).
        """
        return (self.ensemble_size, self.num_time_steps, self.num_channels, self.lmax, self.mmax, 2)

    def _ensure_state(self, batch_size: int, device=None, dtype=None) -> None:
        """Single source of truth for (re)allocating ``self.state``.

        Calling it with the same shape as the current state is a no-op; calling it
        with a different batch size re-registers the buffer so buffer semantics
        (``.to(device)``, ``_buffers`` membership) are preserved rather than
        relying on the ``__setattr__`` hook.
        """
        if device is None:
            device = self.state.device if ("state" in self._buffers) else torch.device("cpu")
        if dtype is None:
            dtype = self.state.dtype if ("state" in self._buffers) else torch.float32

        target_shape = (batch_size,) + tuple(self._state_shape_suffix)
        if ("state" not in self._buffers) or (tuple(self.state.shape) != target_shape):
            self.register_buffer("state", torch.zeros(target_shape, dtype=dtype, device=device), persistent=False)

    def is_stateful(self) -> bool:
        """Whether successive draws depend on the previous state."""
        raise NotImplementedError("is_stateful method not implemented for this noise class")

    def extra_repr(self) -> str:
        return (
            f"num_grid_points={self.num_grid_points}, ensemble_size={self.ensemble_size}, "
            f"num_channels={self.num_channels}, num_time_steps={self.num_time_steps}, "
            f"lmax={self.lmax}, seeds={self.seeds}, reflects={self.reflects}"
        )

    def set_rng(self, seeds: list[int]) -> None:
        r"""Re-seed the per-member private generators.

        The noise stream is deliberately independent of the global RNG, so seeding
        it explicitly is the only way to make a run reproducible or to decorrelate
        ensemble members from one another. One generator per member (rather than a
        single generator drawing the whole block) keeps a member's stream tied to
        its *global* index, so the realisation is invariant to how members are
        distributed over devices.
        """
        self.seeds = list(seeds)
        self._generators: dict[torch.device, list[torch.Generator]] = {}
        self._generators[torch.device("cpu")] = [
            torch.Generator(device="cpu").manual_seed(int(seed)) for seed in self.seeds
        ]

    def _member_generators(self, device: torch.device) -> list[torch.Generator]:
        """Fetch (creating on first use) the per-member generators for ``device``."""
        if device not in self._generators:
            self._generators[device] = [
                torch.Generator(device=device).manual_seed(int(seed)) for seed in self.seeds
            ]
        return self._generators[device]

    def _draw_normal(self, out: torch.Tensor) -> torch.Tensor:
        """Fill ``out`` of shape ``(batch, ensemble, ...)`` with per-member standard normals.

        Reflection is applied per member here, so an antithetic pair is an exact
        element-wise negation of its partner.
        """
        generators = self._member_generators(out.device)
        member_shape = out.shape[:1] + out.shape[2:]
        for member, (generator, reflect) in enumerate(zip(generators, self.reflects)):
            # draw into a contiguous buffer: filling the strided `out[:, member]`
            # view directly is not guaranteed to consume the generator identically
            # across devices and layouts.
            draw = torch.empty(member_shape, dtype=out.dtype, device=out.device)
            draw.normal_(mean=0.0, std=1.0, generator=generator)
            out[:, member] = -draw if reflect else draw
        return out

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Zero the internal state, optionally resizing it.

        Call this between rollouts so a new forecast does not inherit the noise
        trajectory of the previous one.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)
        with torch.no_grad():
            self.state.zero_()

    def update(self, replace_state: bool = False, batch_size: Optional[int] = None) -> None:
        r"""Draw a fresh set of spectral coefficients into the state.

        The base implementation is memoryless: it overwrites the state with new
        standard normal coefficients. Stateful subclasses override this to
        propagate the previous state forward in time instead.

        Parameters
        ----------
        replace_state : bool, optional
            Accepted for interface compatibility with stateful subclasses. Ignored
            here, since the base class always replaces.
        batch_size : int, optional
            If given, resize the state buffer to this batch size before drawing.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            self.state.copy_(self._draw_normal(torch.empty_like(self.state)))

    def get_rng_state(self) -> list[torch.Tensor]:
        """Capture the per-member generator states for checkpointing."""
        device = self.state.device
        return [generator.get_state() for generator in self._member_generators(device)]

    def set_rng_state(self, states: list[torch.Tensor]) -> None:
        """Restore the generators from states captured by :meth:`get_rng_state`."""
        device = self.state.device
        for generator, state in zip(self._member_generators(device), states):
            generator.set_state(state)

    def get_tensor_state(self) -> torch.Tensor:
        """Return a detached copy of the current spectral state."""
        return self.state.detach().clone()

    def set_tensor_state(self, newstate: torch.Tensor) -> None:
        """Overwrite the spectral state, resizing the batch dimension if needed.

        Everything beyond the batch dimension must match the module's expected
        layout. Validated up front so a mismatch reports both layouts rather than
        failing later inside the copy.
        """
        expected_suffix = tuple(self._state_shape_suffix)
        actual_suffix = tuple(newstate.shape[1:]) if newstate.dim() >= 1 else tuple(newstate.shape)
        if actual_suffix != expected_suffix:
            raise ValueError(
                f"set_tensor_state: shape mismatch beyond batch dim. "
                f"Expected suffix {expected_suffix}, got {actual_suffix} "
                f"(full newstate.shape={tuple(newstate.shape)}, current state.shape={tuple(self.state.shape)})."
            )
        if tuple(newstate.shape) != tuple(self.state.shape):
            self._ensure_state(newstate.shape[0])
        with torch.no_grad():
            self.state.copy_(newstate)

    def _transform(self, cstate: torch.Tensor) -> torch.Tensor:
        """Inverse-transform spectral coefficients to the node list.

        Parameters
        ----------
        cstate : torch.Tensor
            Complex coefficients of shape ``(batch, ensemble, time, channels, lmax, mmax)``.

        Returns
        -------
        torch.Tensor
            Real field of shape ``(batch, ensemble, time, channels, points)``.
        """
        with amp.autocast(device_type=cstate.device.type, enabled=False):
            return self.isht(cstate)


class IsotropicGaussianRandomFieldS2(BaseSphericalNoise):
    r"""Isotropic Gaussian random field on the unit sphere. Stateless.

    Draws standard normal spherical harmonic coefficients, scales them by a
    per-degree standard deviation, and transforms to the grid. Because the
    scaling depends only on the degree :math:`l` and not the order :math:`m`, the
    resulting field is statistically isotropic -- no direction or location on the
    sphere is special, which a field generated directly on a lat-lon grid would
    not satisfy.

    The angular power spectrum follows a power law,

    .. math::

        \sigma_l = \sigma \sqrt{\frac{(2l+1)^{-\alpha}}{Z}},
        \qquad
        Z = \sum_l \frac{(2l+1)\,(2l+1)^{-\alpha}}{4\pi}

    where the normalization :math:`Z` fixes the pointwise variance of the field at
    :math:`\sigma^2` regardless of :math:`\alpha` or the truncation, so changing
    the correlation length does not silently change the amplitude.
    :math:`\alpha = 0` gives white noise; larger :math:`\alpha` damps high degrees
    and yields a smoother, longer-correlated field.

    Each call to ``update`` draws an independent realization.

    References
    ----------
    [1] Lang, A.; Schwab C.; Isotropic Gaussian random fields on the sphere:
    regularity, fast simulation and stochastic partial differential equations;
    The Annals of Applied Probability; 2015, Vol. 25, No. 6, 3047-3094;
    DOI: 10.1214/14-AAP1067
    """

    def __init__(self, *, sigma: float = 1.0, alpha: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.sigma = sigma
        self.alpha = float(alpha)

        ls = torch.arange(self.lmax).reshape(-1, 1)
        ms = torch.arange(self.mmax)
        power_spectrum = torch.pow(2 * ls + 1, -self.alpha)
        norm_factor = torch.sum((2 * ls + 1) * power_spectrum / 4.0 / math.pi)
        sigma_l = sigma * torch.sqrt(power_spectrum / norm_factor)
        sigma_l = torch.where(ms <= ls, sigma_l, 0.0)

        # broadcast over (batch, ensemble, time, channels, l, m)
        sigma_l = sigma_l.reshape((1, 1, 1, 1, self.lmax, self.mmax)).to(dtype=torch.float32)

        self.register_buffer("sigma_l", sigma_l * self.basis_scale, persistent=False)

    def is_stateful(self) -> bool:
        return False

    def extra_repr(self) -> str:
        return super().extra_repr() + f", sigma={self.sigma}, alpha={self.alpha}"

    @torch.compiler.disable
    def forward(self, update_internal_state: bool = False) -> torch.Tensor:
        r"""Scale the stored coefficients by the power spectrum and transform to the grid.

        Returns
        -------
        torch.Tensor
            Real-valued noise of shape
            ``(batch, ensemble, num_time_steps, num_channels, num_grid_points)``.
        """
        # torch.view_as_complex on a registered buffer hits a torch.compile/Inductor
        # bug (set_() size mismatch when itemsize changes float32->complex64).
        scaled = self.state / math.sqrt(2)
        cstate = torch.complex(scaled[..., 0], scaled[..., 1]) * self.sigma_l

        eta = self._transform(cstate)

        if update_internal_state:
            self.update()

        return eta


class DiffusionNoiseS2(BaseSphericalNoise):
    r"""Temporally correlated random field from a diffusion process on the sphere. Stateful.

    Perturbations in an ensemble forecast should not be redrawn independently at
    every step -- that produces noise the model damps out immediately rather than
    a coherent perturbation that grows. This module instead evolves the spectral
    coefficients as an Ornstein-Uhlenbeck process, so each step is correlated with
    the last:

    .. math::

        \eta_{t+1} = \phi\, \eta_t + \sqrt{1 - \phi^2}\; \sigma_l\, \xi_t,
        \qquad \phi = e^{-\lambda}

    with :math:`\xi_t` standard normal. The prefactor :math:`\sqrt{1-\phi^2}` keeps
    the process stationary: the marginal variance stays at :math:`\sigma^2` for
    every :math:`t` instead of drifting as the correlation length changes.
    :math:`\lambda = \Delta t / \tau` sets how fast the field decorrelates in time,
    while ``kT`` sets the spatial correlation length through the angular power
    spectrum.

    Both ``kT`` and ``lambd`` may be given per channel, so different variables can
    carry perturbations at different scales. FourCastNet 3 uses eight channels
    whose ``kT`` spans five orders of magnitude.

    Parameters
    ----------
    sigma : float, default is 1.0
        Stationary standard deviation.
    kT : float or list, default is 0.5 * (500 km / 6370 km)^2
        Spatial correlation length. A list must match ``num_channels``.
    lambd : float or list, default is 1.0
        Temporal correlation length, i.e. ``dt / tau``. A list must match
        ``num_channels``.

    References
    ----------
    Palmer, T. et al.; Stochastic parametrization and model uncertainty; ECMWF
    Technical Memorandum 598, 2009, appendix 8.1.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        kT: Union[float, list] = 0.5 * (500.0 / 6370.0) ** 2,
        lambd: Union[float, list] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.sigma = sigma
        self.kT = kT
        self.lambd = lambd

        ls = torch.arange(self.lmax)

        kT = self._as_per_channel_tensor(kT, "kT")
        lambd = self._as_per_channel_tensor(lambd, "lambd")

        ektllp1 = torch.exp(-kT * ls * (ls + 1))
        F0norm = torch.sum((2 * ls[1:] + 1) * ektllp1[..., 1:], dim=-1, keepdim=True)
        phi = torch.exp(-lambd)
        F0 = sigma * torch.sqrt(0.5 * (1 - phi**2) / F0norm)
        sigma_l = F0 * torch.exp(-0.5 * kT * ls * (ls + 1))
        # The ECMWF formulation above is 4-pi normalised; lift it into the
        # orthonormal convention the rest of the derivation assumes.
        sigma_l = math.sqrt(4 * math.pi) * sigma_l

        # broadcast over (batch, ensemble, time, channels, l, m)
        phi = phi.reshape((1, 1, 1, self.num_channels, 1, 1)).to(dtype=torch.float32)
        sigma_l = sigma_l.reshape((1, 1, 1, self.num_channels, self.lmax, 1)).to(dtype=torch.float32)

        # trailing real/imaginary axis of the state buffer
        phi = phi.unsqueeze(-1)
        sigma_l = sigma_l.unsqueeze(-1)

        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("sigma_l", sigma_l * self.basis_scale, persistent=False)

        # For a multi-step history, resampling needs the Toeplitz discount matrix
        #            [    1,     0,   0, 0]
        # discount = [  phi,     1,   0, 0]
        #            [phi^2,   phi,   1, 0]
        #            [phi^3, phi^2, phi, 1]
        if self.num_time_steps > 1:
            discount = []
            for phi_tmp in self.phi.reshape(-1).tolist():
                phivec = np.power(phi_tmp, np.arange(0, self.num_time_steps))
                discount.append(torch.as_tensor(toep(phivec, np.zeros(self.num_time_steps))).to(dtype=torch.float32))
            self.register_buffer("discount", torch.stack(discount, dim=0), persistent=False)

    def _as_per_channel_tensor(self, value: Union[float, list], name: str) -> torch.Tensor:
        # Anything that is not a plain scalar is treated as one entry per channel, so
        # OmegaConf ListConfig and numpy arrays work as well as a Python list.
        if isinstance(value, (int, float)):
            tensor = torch.as_tensor([float(value)]).repeat(self.num_channels)
        else:
            tensor = torch.as_tensor([float(entry) for entry in value], dtype=torch.float32)
            if tensor.shape[0] != self.num_channels:
                raise ValueError(
                    f"expected {name} to have {self.num_channels} entries (one per channel), got {tensor.shape[0]}"
                )
        return tensor.reshape(self.num_channels, 1)

    def is_stateful(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return super().extra_repr() + f", sigma={self.sigma}, kT={self.kT}, lambd={self.lambd}"

    def update(self, replace_state: bool = False, batch_size: Optional[int] = None) -> None:
        r"""Advance the noise process by one step, or resample the whole history.

        Parameters
        ----------
        replace_state : bool, optional
            If ``False`` (the default), take one autoregressive step: the existing
            state is damped by :math:`\phi` and a fresh innovation is added, so the
            new field stays correlated with the old one. When ``num_time_steps > 1``
            the oldest step is dropped and the new one appended.

            If ``True``, discard the state and draw a fresh history from the
            *stationary* distribution. The first time step is scaled by
            :math:`1/\sqrt{1-\phi^2}` and, for ``num_time_steps > 1``, the history
            is correlated by a Toeplitz discount matrix of powers of :math:`\phi`.
            This matters when starting a rollout: drawing independent steps instead
            would begin from a field with the wrong variance and no temporal
            structure, and the process would need many steps to spin up.
        batch_size : int, optional
            If given, resize the state buffer to this batch size before drawing.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            with amp.autocast(device_type=self.state.device.type, enabled=False):
                if replace_state:
                    eta_l = torch.empty_like(self.state)
                else:
                    batch = self.state.shape[0]
                    eta_l = torch.empty(
                        (batch, self.ensemble_size, 1, self.num_channels, self.lmax, self.mmax, 2),
                        dtype=self.state.dtype,
                        device=self.state.device,
                    )
                eta_l = self._draw_normal(eta_l)
                eta_l = self.sigma_l * eta_l

                if not replace_state:
                    if self.num_time_steps > 1:
                        last_state = self.state[:, :, -1, ...].unsqueeze(2)
                        newstep = self.phi * last_state + eta_l
                        newstate = torch.cat([self.state[:, :, 1:, ...], newstep], dim=2)
                    else:
                        newstate = self.phi * self.state + eta_l
                else:
                    newstate = eta_l
                    # the first element of the history needs a different weighting to
                    # sample the stationary distribution
                    newstate[:, :, 0, ...] = newstate[:, :, 0, ...] / torch.sqrt(1.0 - self.phi**2)
                    if self.num_time_steps > 1:
                        newstate = torch.einsum("ctr,berclmu->betclmu", self.discount, newstate).contiguous()

                self.state.copy_(newstate)

    @torch.compiler.disable
    def forward(self, update_internal_state: bool = False) -> torch.Tensor:
        r"""Transform the current spectral state to a noise field on the node list.

        Returns
        -------
        torch.Tensor
            Real-valued noise of shape
            ``(batch, ensemble, num_time_steps, num_channels, num_grid_points)``.
        """
        cstate = torch.complex(self.state[..., 0], self.state[..., 1])

        eta = self._transform(cstate)

        if update_internal_state:
            self.update()

        return eta


class DummyNoiseS2(BaseSphericalNoise):
    r"""Dummy noise module for testing and debugging. Stateless.

    Always emits a tensor with the correct output shape but carries no stochastic
    signal beyond what the chosen mode specifies. Unlike the real noise classes it
    stores its state in *spatial* rather than spectral layout and never runs an
    SHT, so it is cheap enough for tests that only care about shapes and control
    flow.

    Parameters
    ----------
    mode : str, default 'constant_zero'
        ``'constant_zero'`` always emits zeros, which leaves the model input
        untouched so integration tests can check shapes without accounting for
        random perturbations. ``'constant_random'`` draws a Gaussian tensor once
        per :meth:`update` and holds it fixed until the next one.
    """

    def __init__(self, *, mode: str = "constant_zero", **kwargs) -> None:
        if mode not in ("constant_zero", "constant_random"):
            raise ValueError(f"DummyNoiseS2: unknown mode '{mode}'. Expected 'constant_zero' or 'constant_random'.")
        self.mode = mode
        super().__init__(**kwargs)

    @property
    def _state_shape_suffix(self) -> tuple[int, ...]:
        # spatial (points,) rather than spectral (l, m, 2)
        return (self.ensemble_size, self.num_time_steps, self.num_channels, self.num_grid_points)

    def is_stateful(self) -> bool:
        return False

    def extra_repr(self) -> str:
        return super().extra_repr() + f", mode={self.mode}"

    def update(self, replace_state: bool = False, batch_size: Optional[int] = None) -> None:
        """Refresh the stored tensor according to the selected mode."""
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            newstate = torch.empty_like(self.state)
            if self.mode == "constant_zero":
                newstate.zero_()
            else:
                newstate = self._draw_normal(newstate)
            self.state.copy_(newstate)

    def forward(self, update_internal_state: bool = False) -> torch.Tensor:
        """Return the stored tensor; no transform involved."""
        state = self.state

        if update_internal_state:
            self.update()

        return state


def noise_seeds_reflects(
    ensemble_size: int,
    *,
    centered: bool,
    member_offset: int = 0,
    num_members_total: Optional[int] = None,
    group_id: int = 0,
    base_seed: int = 333,
    seed_offset: int = 0,
) -> tuple[list[int], list[bool]]:
    r"""Derive per-member seeds and reflection flags from *global* member indices.

    Keying off the global index -- rather than the position within the local
    tensor -- is what makes a realisation invariant to how members are spread over
    devices, so changing ``ensemble_size_per_device`` or ``num_gpus_per_ensemble``
    reshuffles nothing.

    ``centered=False``: every member is fully independent.

    ``centered=True``: antithetic pairing -- global members (0,1), (2,3), ... share
    a seed and differ only by a sign flip, which cancels first-order sampling error
    in the ensemble estimator.

    Parameters
    ----------
    ensemble_size : int
        Number of members held locally.
    centered : bool
        Enable antithetic pairing.
    member_offset : int, optional
        Global index of the first local member.
    num_members_total : int, optional
        Total members across the whole ensemble; sets the stride between
        data-parallel groups. Defaults to ``member_offset + ensemble_size``.
    group_id : int, optional
        Index of the ensemble communication group, i.e. which data-parallel slice
        this rank belongs to. Decorrelates concurrent batches.
    base_seed, seed_offset : int, optional
        Fixed offsets. ``seed_offset`` lets independent noise sources draw from
        decorrelated streams while keeping the same per-member structure.
    """
    num_members_total = num_members_total or (member_offset + ensemble_size)

    seeds, reflects = [], []
    for member in range(ensemble_size):
        global_index = member_offset + member
        if centered:
            key = global_index // 2
            reflect = global_index % 2 == 0
        else:
            key = global_index
            reflect = False
        seeds.append(base_seed + seed_offset + key + num_members_total * group_id)
        reflects.append(reflect)

    return seeds, reflects


def build_noise(
    noise_params: dict,
    *,
    batch_size: int,
    ensemble_size: int,
    num_channels: int,
    num_time_steps: int,
    seeds: list[int],
    reflects: list[bool],
    grid: Optional[Union[str, int]] = None,
    transform: Optional[tuple[InverseSpectralTransform, int, int]] = None,
    default_lambd: float = 1.0,
) -> BaseSphericalNoise:
    """Construct a noise module from a config dict.

    ``default_lambd`` supplies the temporal correlation default (``dt / 6h`` in
    FourCastNet 3) since it depends on the dataset cadence.

    Pass ``transform`` (the triple returned by :func:`build_inverse_sht`) to reuse
    an existing transform. Building one is expensive -- the Legendre basis is
    ``O(lmax^2 * nlat)`` -- so callers that re-create the noise module, e.g. when
    the ensemble layout changes, should never rebuild it.
    """
    noise_type = noise_params.get("type", None)
    if noise_type is None:
        raise ValueError("Error, please specify a noise type")

    if transform is None:
        if grid is None:
            raise ValueError("build_noise requires either 'grid' or a prebuilt 'transform'.")
        transform = build_inverse_sht(
            grid,
            lmax=noise_params.get("lmax", None),
            use_graphed_irfft=noise_params.get("use_graphed_irfft", False),
        )
    isht, lmax, num_grid_points = transform

    common = {
        "isht": isht,
        "lmax": lmax,
        "num_grid_points": num_grid_points,
        "batch_size": batch_size,
        "ensemble_size": ensemble_size,
        "num_channels": num_channels,
        "num_time_steps": num_time_steps,
        "seeds": seeds,
        "reflects": reflects,
    }

    if noise_type == "diffusion":
        return DiffusionNoiseS2(
            sigma=noise_params.get("sigma", 1.0),
            kT=noise_params.get("kT", 0.5 * (100 / 6370) ** 2),
            lambd=noise_params.get("lambd", default_lambd),
            **common,
        )
    if noise_type == "white":
        return IsotropicGaussianRandomFieldS2(
            sigma=noise_params.get("sigma", 1.0),
            alpha=noise_params.get("alpha", 0.0),
            **common,
        )
    if noise_type == "dummy":
        return DummyNoiseS2(mode=noise_params.get("mode", "constant_zero"), **common)

    raise NotImplementedError(f"Error, noise type {noise_type} not supported.")

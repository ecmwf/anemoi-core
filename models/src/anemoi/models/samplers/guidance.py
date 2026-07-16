# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diffusion guidance utilities for EDM-style sampling.

This module provides :class:`GuidedDenoiser`, a thin wrapper that combines two
denoisers -- a *main* model ``D1`` and a *guide* model ``D0`` -- into a single
denoising callable that the existing samplers can consume **unchanged**.

Linear guidance in denoised (``D``) space, following Karras et al. 2024
("Guiding a Diffusion Model with a Bad Version of Itself", Eq. 3)::

    D_w(x; sigma) = D1(x; sigma) + (w - 1) * (D1(x; sigma) - D0(x; sigma))

The same formula and the same wrapper cover two regimes:

* **Autoguidance** -- ``D0`` is a *degraded copy of the main model* (an earlier
  checkpoint and/or smaller capacity) trained on the *same data pipeline*. This
  improves sample quality while preserving ensemble spread. Use the full sigma
  range (the default), i.e. no interval gating.
* **Classifier-free-style guidance** -- ``D0`` is a different (e.g.
  unconditional) model. Here one typically restricts guidance to a mid-sigma
  band via ``sigma_min`` / ``sigma_max`` (the "guidance interval").

The wrapper is intentionally signature-agnostic: it forwards *all* positional
and keyword arguments unchanged to both denoisers, so it works for any denoiser
matching the :data:`~anemoi.models.samplers.diffusion_samplers.DenoisingFunction`
contract (forecaster and downscaler alike). It only needs ``sigma`` for interval
gating, which by that contract is always the third-from-last positional
argument: ``(..., sigma, model_comm_group, grid_shard_shapes)``.

Correctness contract for the difference ``D1 - D0`` to be meaningful:

* both denoisers must consume the *same* (normalised) conditioning inputs, and
* both must produce outputs in the *same* normalised space.

Each denoiser applies its own EDM preconditioning (its own ``sigma_data``)
internally, so guides trained with a different ``sigma_data`` are fine as long
as the input/output normalisation matches.
"""

from __future__ import annotations

import logging

import torch

from anemoi.models.samplers.diffusion_samplers import DenoisingFunction

LOGGER = logging.getLogger(__name__)

# By the DenoisingFunction contract the call signature ends with
# (..., sigma, model_comm_group, grid_shard_shapes), so sigma is at index -3.
SIGMA_ARG_INDEX = -3


class GuidedDenoiser:
    """Combine a main and a guide denoiser into one guided denoising callable.

    Instances are drop-in replacements for a model's ``fwd_with_preconditioning``
    method: pass one to ``sampler.sample(...)`` in place of the plain denoiser.

    Parameters
    ----------
    main_denoiser : DenoisingFunction
        The main model's ``fwd_with_preconditioning`` (``D1``).
    guide_denoiser : DenoisingFunction
        The guide model's ``fwd_with_preconditioning`` (``D0``). Must satisfy the
        correctness contract described in the module docstring.
    weight : float
        Guidance weight ``w``. ``w == 1.0`` reduces to the main denoiser, with no
        extra forward pass (guidance disabled).
    sigma_min : float, optional
        Lower bound of the guidance interval. Default ``0.0`` (no lower gating).
    sigma_max : float, optional
        Upper bound of the guidance interval. Default ``inf`` (no upper gating).
        With both defaults, guidance is applied at every sigma -- the recommended
        setting for autoguidance.
    sigma_arg_index : int, optional
        Position of the ``sigma`` argument in the call. Defaults to
        :data:`SIGMA_ARG_INDEX` (``-3``), matching the ``DenoisingFunction``
        contract.
    """

    def __init__(
        self,
        main_denoiser: DenoisingFunction,
        guide_denoiser: DenoisingFunction,
        weight: float,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_arg_index: int = SIGMA_ARG_INDEX,
    ) -> None:
        self.main_denoiser = main_denoiser
        self.guide_denoiser = guide_denoiser
        self.weight = float(weight)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_arg_index = sigma_arg_index

    @property
    def interval_active(self) -> bool:
        """Whether a non-trivial guidance interval (sigma gating) is configured."""
        return self.sigma_min > 0.0 or self.sigma_max < float("inf")

    def _sigma_in_interval(self, sigma: torch.Tensor) -> bool:
        # sigma is broadcast but constant across batch/ensemble at each sampling
        # step, so a single scalar read is sufficient for gating.
        s = float(sigma.flatten()[0])
        return self.sigma_min <= s <= self.sigma_max

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        d1 = self.main_denoiser(*args, **kwargs)

        # w == 1 is a pure no-op: skip the guide forward pass entirely.
        if self.weight == 1.0:
            return d1

        # Optional guidance interval (off by default): outside the band, return
        # the un-guided prediction and skip the guide forward pass.
        if self.interval_active and not self._sigma_in_interval(args[self.sigma_arg_index]):
            return d1

        d0 = self.guide_denoiser(*args, **kwargs)
        return d1 + (self.weight - 1.0) * (d1 - d0)

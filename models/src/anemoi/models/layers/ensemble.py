# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from abc import ABC
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from typing import Union

import einops
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.noise_modulation import GroupedStdModulation
from anemoi.models.layers.noise_modulation import climatological_spread_statistics
from anemoi.models.layers.noise_modulation import resolve_prefixed_variables
from anemoi.models.layers.noise_modulation import strip_level
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter
from anemoi.models.layers.spherical_noise import EARTH_RADIUS_KM
from anemoi.models.layers.spherical_noise import BaseSphericalNoise
from anemoi.models.layers.spherical_noise import DiffusionNoiseS2
from anemoi.models.layers.spherical_noise import band_limit
from anemoi.models.layers.spherical_noise import build_inverse_sht
from anemoi.models.layers.spherical_noise import build_noise
from anemoi.models.layers.spherical_noise import degree_variance
from anemoi.models.layers.spherical_noise import filter_truncation
from anemoi.models.layers.spherical_noise import heat_kernel_response
from anemoi.models.layers.spherical_noise import lowpass_response
from anemoi.models.layers.spherical_noise import noise_seeds_reflects
from anemoi.models.layers.spherical_noise import tabulated_coefficient_variance
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def _to_container(config):
    """Hydra hands over OmegaConf containers; resolve to plain Python once, at the boundary."""
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def kT_from_length_scale(length_km: float) -> float:
    r"""FourCastNet 3's ``kT`` for a correlation length scale: :math:`kT = \tfrac12 (L / a)^2`."""
    return 0.5 * (float(length_km) / EARTH_RADIUS_KM) ** 2


@dataclass
class NoiseChannel:
    """One input-noise channel: its angular power spectrum and, optionally, the spread that scales it.

    Parameters
    ----------
    name : str
        Channel name, used in logs and as the key in the ``channels`` config.
    kT : float, optional
        FourCastNet 3 spectrum, :math:`\\sigma_l^2 \\propto e^{-kT\\,l(l+1)}`.
    spectrum : dict, optional
        Tabulated spectrum, ``{"degree": [...], "sigma2": [...]}``, see
        :func:`~anemoi.models.layers.spherical_noise.tabulated_coefficient_variance`.
        Exactly one of ``kT`` and ``spectrum``.
    spread : tuple[str, ...]
        Spread variables whose map scales the channel, e.g. ``("u_850", "v_850")``,
        matched like :class:`~anemoi.models.layers.noise_modulation.GroupedStdModulation`
        groups. Empty for a channel that is never scaled.
    smoothing_km : float, optional
        Smoothing length scale of the channel's spread map, overriding the
        ``modulation`` rule.
    """

    name: str
    kT: Optional[float] = None
    spectrum: Optional[dict] = None
    spread: tuple[str, ...] = ()
    smoothing_km: Optional[float] = None

    KEYS = ("kT", "spectrum", "spread", "smoothing_km")

    @classmethod
    def from_config(cls, name: str, config: dict) -> "NoiseChannel":
        config = dict(config or {})
        unknown = sorted(set(config) - set(cls.KEYS))
        if unknown:
            raise ValueError(f"input noise channel '{name}' has unknown keys {unknown}; expected {list(cls.KEYS)}.")
        if (config.get("kT") is None) == (config.get("spectrum") is None):
            raise ValueError(f"input noise channel '{name}' needs exactly one of 'kT' and 'spectrum'.")
        spectrum = config.get("spectrum")
        if spectrum is not None:
            spectrum = {"degree": list(spectrum["degree"]), "sigma2": list(spectrum["sigma2"])}
        smoothing_km = config.get("smoothing_km")
        return cls(
            name=name,
            kT=float(config["kT"]) if config.get("kT") is not None else None,
            spectrum=spectrum,
            spread=tuple(config.get("spread") or ()),
            smoothing_km=float(smoothing_km) if smoothing_km is not None else None,
        )

    def coefficient_variance(self, lmax: int) -> Tensor:
        """Per-degree coefficient variance, float64 of shape ``(lmax,)``."""
        if self.kT is not None:
            return heat_kernel_response(torch.tensor([self.kT]), lmax)[0]
        return tabulated_coefficient_variance(self.spectrum["degree"], self.spectrum["sigma2"], lmax)


def _contiguous_blocks(indices: list[int], max_size: int) -> list[tuple[slice, slice]]:
    """Split sorted ``indices`` into runs of consecutive values of at most ``max_size``.

    Returns ``(slice into the values, slice into the list)`` pairs, so a block can
    address both the noise channels and their position among ``indices``.
    """
    blocks = []
    start = 0
    for position in range(1, len(indices) + 1):
        run_ends = position == len(indices) or indices[position] != indices[position - 1] + 1
        if run_ends or position - start == max_size:
            blocks.append((slice(indices[start], indices[position - 1] + 1), slice(start, position)))
            start = position
    return blocks


class SphericalInputNoise(nn.Module):
    r"""FourCastNet 3 style input perturbation.

    Unlike :class:`NoiseConditioning`, which draws white noise on the hidden grid
    and uses it to condition the processor, this module reproduces FourCastNet 3:
    a spatially (and optionally temporally) correlated random field is generated
    in spectral space, inverse-transformed onto the *data* grid, and concatenated
    to the model input as extra channels -- one set per input time step. The
    network itself is left untouched; FourCastNet 3 injects no noise internally.

    The underlying field is built lazily, on the first :meth:`advance`, because
    the ensemble layout (how many members this rank holds, and their global
    indices) is only known once a batch arrives. The spherical harmonic basis is
    built eagerly here and reused, since it is by far the expensive part.

    The channels are given either as FourCastNet 3 does -- ``n_channels`` and one
    ``noise.kT`` per channel -- or as a named ``channels`` mapping, where each channel
    has its own spectrum: ``kT``, or a tabulated ``spectrum`` (see :class:`NoiseChannel`).

    Parameters
    ----------
    grid : str or int
        Grid the data nodes live on, e.g. ``"n320"``. See
        :func:`~anemoi.models.layers.spherical_noise.build_inverse_sht`.
    noise : dict
        Noise field configuration: ``type`` (``diffusion`` / ``white`` / ``dummy``)
        plus the parameters of that type (``sigma``, ``kT``, ``lambd``, ``alpha``,
        ``lmax``).
    n_channels : int, optional
        Number of noise channels appended per input time step. FourCastNet 3 uses
        eight, each with a different spatial correlation length. Defaults to one,
        or to the number of ``channels``.
    channels : dict, optional
        Channel name to its spectrum, ``{kT: ...}`` or ``{spectrum: {degree, sigma2}}``.
        Needs a ``diffusion`` field, and replaces ``noise.kT``.
    centered : bool, optional
        Antithetic pairing of ensemble members. FourCastNet 3 pretrains with
        ``False`` and fine-tunes with ``True``.
    dataset : str, optional
        Name of the dataset whose input the noise is appended to. Required when
        the model has more than one input dataset.
    default_lambd : float, optional
        Temporal decorrelation default, ``dt / 6h`` in FourCastNet 3.
    num_time_steps : int
        Supplied by the model: ``multistep_input``, mirroring makani's
        ``n_history + 1``.
    num_grid_points : int, optional
        Supplied by the model. Checked against the transform so a mismatched grid
        fails at construction rather than producing a silently misaligned field.
    name_to_index : dict, optional
        Supplied by the model, for noise that reads the model input
        (:class:`SphericalInputConditionedNoise`). Unused here.
    statistics : dict, optional
        Supplied by the model, as ``name_to_index``. Unused here.
    name_to_index_stats : dict, optional
        Supplied by the model, as ``name_to_index``. Unused here.
    """

    # Whether channels may name spread variables; only the conditioned subclass reads them.
    _reads_spread = False

    def __init__(
        self,
        *,
        grid: Union[str, int],
        noise: dict,
        n_channels: Optional[int] = None,
        channels: Optional[dict] = None,
        centered: bool = False,
        dataset: Optional[str] = None,
        default_lambd: float = 1.0,
        num_time_steps: int = 1,
        num_grid_points: Optional[int] = None,
        name_to_index: Optional[dict[str, int]] = None,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()

        self.noise_params = dict(_to_container(noise))
        self.channels = self._parse_channels(_to_container(channels), n_channels)
        self.n_channels = len(self.channels) if self.channels else (n_channels or 1)
        self.centered = centered
        self.dataset = dataset
        self.default_lambd = default_lambd
        self.num_time_steps = num_time_steps

        self._transform = build_inverse_sht(
            grid,
            lmax=self.noise_params.get("lmax", None),
            use_graphed_irfft=self.noise_params.get("use_graphed_irfft", False),
        )
        _, self.lmax, self.num_grid_points = self._transform

        if num_grid_points is not None and num_grid_points != self.num_grid_points:
            raise ValueError(
                f"SphericalInputNoise: grid '{grid}' has {self.num_grid_points} points but the "
                f"'{dataset}' nodes have {num_grid_points}. The noise field would not align with "
                "the data nodes."
            )

        # Per-channel spectrum of named channels; None keeps the noise's own kT formula.
        self._coefficient_variance: Optional[Tensor] = None
        if self.channels:
            self._coefficient_variance = torch.stack(
                [channel.coefficient_variance(self.lmax) for channel in self.channels]
            )

        self.noise: Optional[BaseSphericalNoise] = None
        self._layout = None
        self._autoregressive_seen = False

        LOGGER.info(
            "SphericalInputNoise: type=%s, %d channels x %d time steps on %s (%d points), lmax=%d, centered=%s",
            self.noise_params.get("type"),
            self.n_channels,
            self.num_time_steps,
            grid,
            self.num_grid_points,
            self.lmax,
            self.centered,
        )
        if self.channels:
            kinds = Counter("kT" if channel.kT is not None else "tabulated" for channel in self.channels)
            LOGGER.info("SphericalInputNoise: channel spectra %s", dict(kinds))

    def _parse_channels(self, channels: Optional[dict], n_channels: Optional[int]) -> list[NoiseChannel]:
        """Named channels from the config, validated against the rest of it."""
        if not channels:
            return []
        if self.noise_params.get("type") != "diffusion":
            raise ValueError(
                f"input noise 'channels' set their own spectra, which needs a 'diffusion' noise field; "
                f"got type '{self.noise_params.get('type')}'."
            )
        if "kT" in self.noise_params:
            raise ValueError("input noise: give 'kT' per channel under 'channels', not under 'noise'.")
        if n_channels is not None and n_channels != len(channels):
            raise ValueError(f"input noise: n_channels={n_channels} but {len(channels)} channels are defined.")
        parsed = [NoiseChannel.from_config(name, config) for name, config in channels.items()]
        if not self._reads_spread and any(channel.spread for channel in parsed):
            raise ValueError(
                f"{type(self).__name__} does not read spread fields; channels naming 'spread' need "
                "SphericalInputConditionedNoise."
            )
        return parsed

    @staticmethod
    def _resolve_base_seed(seed: Optional[int]) -> tuple[int, str]:
        """The seed every member's noise stream is derived from, and where it came from.

        Training passes the run's seed. Inference passes none: then the noise follows
        torch's own seed, so runs seeded differently -- e.g. the members of an
        ensemble forecast -- draw different perturbations, and a repeated seed
        reproduces one.
        """
        if seed is not None:
            return int(seed) % 2**32, "the run seed"
        return int(torch.initial_seed()) % 2**32, "torch.initial_seed()"

    @property
    def consumed_input_idx(self) -> list[int]:
        """Positions in the dataset's model input that this module reads in place of the encoder.

        The model drops these channels from the encoder input. The homogeneous
        perturbation reads nothing.
        """
        return []

    @property
    def conditioned(self) -> bool:
        """Whether :meth:`advance` needs the consumed input channels at ``fcstep == 0``."""
        return False

    def advance(
        self,
        *,
        fcstep: int,
        batch_size: int,
        ensemble_size: int,
        member_offset: int = 0,
        num_members_total: Optional[int] = None,
        group_id: int = 0,
        device: Optional[torch.device] = None,
        inputs: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        r"""Step the noise process forward, ready for the next :meth:`sample`.

        ``fcstep == 0`` starts a new rollout and redraws the history from the
        stationary distribution; later steps take a single autoregressive step so
        the perturbation stays correlated with the previous one. This mirrors
        makani's stepper, which passes ``replace_state=True`` only on the first
        step of a rollout.

        ``member_offset``, ``num_members_total`` and ``group_id`` describe where
        this rank's members sit in the *global* ensemble, so the realisation does
        not depend on how members happen to be distributed over devices.

        ``inputs`` carries the :attr:`consumed_input_idx` channels for subclasses
        that condition on them; the homogeneous perturbation ignores it.

        ``seed`` is the base seed of the noise streams (see :meth:`_resolve_base_seed`).
        """
        base_seed, seed_source = self._resolve_base_seed(seed)
        layout = (ensemble_size, member_offset, num_members_total, group_id, base_seed)
        if self.noise is None or layout != self._layout:
            seeds, reflects = noise_seeds_reflects(
                ensemble_size,
                centered=self.centered,
                member_offset=member_offset,
                num_members_total=num_members_total,
                group_id=group_id,
                base_seed=base_seed,
            )
            self.noise = build_noise(
                self.noise_params,
                transform=self._transform,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                num_channels=self.n_channels,
                num_time_steps=self.num_time_steps,
                seeds=seeds,
                reflects=reflects,
                default_lambd=self.default_lambd,
                coefficient_variance=self._coefficient_variance,
            ).to(device)
            if self._layout is None:
                LOGGER.info("SphericalInputNoise: noise base seed %d, from %s.", base_seed, seed_source)
            self._layout = layout
            LOGGER.debug("SphericalInputNoise: built field with seeds=%s reflects=%s", seeds, reflects)

        if fcstep > 0 and not self._autoregressive_seen:
            # Only reachable with rollout > 1; worth one line of evidence that it ran.
            self._autoregressive_seen = True
            LOGGER.info(
                "SphericalInputNoise: first autoregressive step (fcstep=%d), "
                "advancing the noise trajectory with phi=exp(-lambd)",
                fcstep,
            )

        self.noise.update(replace_state=(fcstep == 0), batch_size=batch_size)

    def sample(self) -> Tensor:
        """Current noise field, shape ``(batch, ensemble, time, channels, points)``."""
        if self.noise is None:
            raise RuntimeError("SphericalInputNoise.sample() called before advance().")
        return self.noise()


class SphericalInputConditionedNoise(SphericalInputNoise):
    r"""FourCastNet 3 input noise, channel by channel scaled by the analysis spread.

    :class:`SphericalInputNoise` injects noise of the same amplitude everywhere and
    at all times. Here a channel may name ``spread`` variables -- ensemble standard
    deviations read from the model input, e.g. the ERA5 EDA spread joined in as
    ``std_``-prefixed forcings -- whose map then scales it, so the channel is strong
    where and when the initial conditions are uncertain and weak where they are
    not. Every channel keeps its own spectrum (``kT`` or tabulated, see
    :class:`NoiseChannel`) and the Ornstein-Uhlenbeck process in time. A typical
    configuration gives each spread variable a channel of its own, with that
    variable's own spread spectrum, plus a few unscaled large-scale FourCastNet 3
    channels.

    At the start of every rollout (``fcstep == 0``):

    1. :class:`~anemoi.models.layers.noise_modulation.GroupedStdModulation` turns the
       spread fields of each history step into one multiplier :math:`M_c` per scaled
       channel: each spread field over a reference (``modulation.reference``),
       averaged over the channel's variables, smoothed, clipped and rescaled.
    2. The freshly drawn stationary noise :math:`\eta_c` is multiplied by :math:`M_c`.
    3. Multiplying in grid space convolves the two spectra, which moves energy out
       of the channel's band, mostly to smaller scales. Each product is therefore
       low-pass filtered back to its own band (:func:`lowpass_response`, from the
       channel's own spectrum) and, with ``band_filter.preserve_variance``, rescaled
       to its variance before filtering: the filter moves energy back into the band
       rather than discarding it.
    4. The result is written back into the spectral state of the noise process.

    Later rollout steps need no spread fields: the unchanged FourCastNet 3 update
    :math:`\eta \leftarrow \phi\,\eta + \sqrt{1-\phi^2}\,\sigma_l\,\xi` carries the
    scaled field forward, so the initial-condition structure decays by
    :math:`\phi` per step while fresh unscaled noise takes over.

    All ``variable_prefix`` inputs are dropped from the encoder input by the model,
    so they shape the noise without becoming input features. With
    ``modulation.enabled: False`` they are still dropped and every channel is left
    unscaled: a baseline with the same channels and the same encoder inputs.

    Parameters
    ----------
    channels : dict
        Channel name to ``{kT | spectrum, spread, smoothing_km}``, see :class:`NoiseChannel`.
    modulation : dict, optional
        How the spread scales the channels:

        - ``enabled`` (``True``); ``variable_prefix`` (``"std_"``); ``source`` (``"eda_stdev"``);
        - ``reference``: ``climatology`` (each spread variable's dataset mean, which
          keeps how uncertain the whole analysis is) or ``sample_mean`` (each
          sample's own global mean, which keeps only where it is uncertain);
        - ``normalizer``: how the data config normalises the spread variables,
          ``std``, ``max`` or ``none``; needed by ``climatology``;
        - ``rescale``: ``climatology`` (a fixed constant per channel, giving the
          multiplier a long-term mean square of about one), ``sample_rms`` (unit
          mean square on every sample) or ``none``; unset, it follows ``reference``;
        - smoothing of each multiplier: ``smoothing_km`` (default length scale,
          ``100``; ``None`` for none), ``smoothing_km_by_variable`` (per base
          variable; a channel takes the coarsest of its variables), or
          ``smooth_to_channel`` (each ``kT`` channel at its own scale);
          a channel's own ``smoothing_km`` overrides all of them;
        - ``clip`` (``(0.05, 10.0)``, in multiples of the reference); ``area_weight``;
        - ``band_filter``: ``quantile``, ``taper``, ``preserve_variance``.
    name_to_index : dict
        Supplied by the model: the dataset's model-input name to position map.
    statistics : dict, optional
        Supplied by the model: the dataset statistics, needed by ``reference: climatology``.
    name_to_index_stats : dict, optional
        Supplied by the model: positions in ``statistics``.
    **kwargs
        Passed to :class:`SphericalInputNoise`.
    """

    _reads_spread = True

    MODULATION_KEYS = (
        "enabled",
        "variable_prefix",
        "source",
        "reference",
        "normalizer",
        "rescale",
        "smoothing_km",
        "smoothing_km_by_variable",
        "smooth_to_channel",
        "clip",
        "area_weight",
        "band_filter",
    )
    REFERENCES = ("climatology", "sample_mean")
    RESCALINGS = ("climatology", "sample_rms", "none")

    def __init__(
        self,
        *,
        channels: dict,
        modulation: Optional[dict] = None,
        name_to_index: Optional[dict[str, int]] = None,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if not channels:
            raise ValueError("SphericalInputConditionedNoise needs named 'channels'.")
        super().__init__(channels=channels, **kwargs)

        if name_to_index is None:
            raise ValueError("SphericalInputConditionedNoise needs name_to_index, which the model supplies.")
        modulation = dict(_to_container(modulation) or {})
        unknown = sorted(set(modulation) - set(self.MODULATION_KEYS))
        if unknown:
            raise ValueError(f"modulation has unknown keys {unknown}; expected {list(self.MODULATION_KEYS)}.")

        self.enabled = bool(modulation.get("enabled", True))
        self.variable_prefix = modulation.get("variable_prefix", "std_")
        source = modulation.get("source", "eda_stdev")
        if source != "eda_stdev":
            raise ValueError(f"modulation.source '{source}' is not supported; expected 'eda_stdev'.")

        spread_inputs = resolve_prefixed_variables(name_to_index, self.variable_prefix)
        self._consumed_input_idx = [index for index, _ in spread_inputs]
        self.std_modulation: Optional[GroupedStdModulation] = None
        self.spectral_filter: Optional[SphericalSpectralFilter] = None
        self.modulated_channels: list[int] = []
        self._modulated_blocks: list[tuple[slice, slice]] = []
        self._modulation_seen = False

        if not self.enabled:
            LOGGER.info(
                "SphericalInputConditionedNoise: modulation disabled; %d '%s' inputs are dropped from the encoder "
                "input and all %d channels are left unscaled.",
                len(self._consumed_input_idx),
                self.variable_prefix,
                self.n_channels,
            )
            return

        self.modulated_channels = [index for index, channel in enumerate(self.channels) if channel.spread]
        if not self.modulated_channels:
            raise ValueError("No channel names 'spread' variables: give some, or set modulation.enabled: False.")
        chunk = DiffusionNoiseS2.transform_chunk_channels or len(self.modulated_channels)
        self._modulated_blocks = _contiguous_blocks(self.modulated_channels, chunk)

        # One spread group per distinct list of variables; channels sharing a list share its map.
        group_names: dict[tuple[str, ...], str] = {}
        channel_group = [
            group_names.setdefault(self.channels[index].spread, self.channels[index].name)
            for index in self.modulated_channels
        ]
        groups = {name: list(spread) for spread, name in group_names.items()}

        reference, mean_square, reference_mode, rescale = self._spread_reference(
            modulation, [name for _, name in spread_inputs], statistics, name_to_index_stats
        )

        smoothing_kT = self._smoothing_kT(modulation)
        band = dict(modulation.get("band_filter") or {})
        quantile, taper = float(band.get("quantile", 0.99)), float(band.get("taper", 0.25))
        self.preserve_filtered_variance = bool(band.get("preserve_variance", True))
        variance = degree_variance(self._coefficient_variance[self.modulated_channels])
        band_response = lowpass_response(variance, quantile=quantile, taper=taper)
        smooth_response = None
        if any(kT is not None for kT in smoothing_kT):
            smooth_response = torch.stack(
                [
                    (
                        heat_kernel_response(torch.tensor([kT]), self.lmax)[0]
                        if kT is not None
                        else torch.ones(self.lmax, dtype=torch.float64)
                    )
                    for kT in smoothing_kT
                ]
            )

        # The filter only needs the degrees its responses reach, far fewer than lmax.
        responses = [band_response] + ([smooth_response] if smooth_response is not None else [])
        truncation = filter_truncation(responses, self.lmax)
        lons_per_lat = self._transform[0].lons_per_lat
        self.spectral_filter = SphericalSpectralFilter(lons_per_lat, truncation)
        self.register_buffer("band_response", band_response[:, : truncation + 1].to(torch.float32), persistent=False)

        if modulation.get("area_weight", True):
            area_weights = torch.as_tensor(quadrature_weights(lons_per_lat))
        else:
            area_weights = torch.full((self.num_grid_points,), 1.0 / self.num_grid_points)

        self.std_modulation = GroupedStdModulation(
            groups=groups,
            channel_group=channel_group,
            name_to_index=name_to_index,
            area_weights=area_weights,
            variable_prefix=self.variable_prefix,
            reference=reference,
            mean_square=mean_square,
            clip=modulation.get("clip", (0.05, 10.0)),
            preserve_total_variance=rescale == "sample_rms",
            spectral_filter=self.spectral_filter if smooth_response is not None else None,
            smooth_response=smooth_response[:, : truncation + 1] if smooth_response is not None else None,
        )
        self._log_layout(reference_mode, modulation.get("normalizer"), rescale, smoothing_kT, variance, quantile)

    def _spread_reference(
        self,
        modulation: dict,
        spread_variables: list[str],
        statistics: Optional[dict],
        name_to_index_stats: Optional[dict[str, int]],
    ) -> tuple[Optional[Tensor], Optional[Tensor], str, str]:
        """Reference and long-term mean square of every spread input, for the configured modes."""
        reference_mode = modulation.get("reference", "climatology")
        if reference_mode not in self.REFERENCES:
            raise ValueError(f"modulation.reference must be one of {self.REFERENCES}, got '{reference_mode}'.")
        # Unset rescaling follows the reference; "none" switches it off.
        rescale = modulation.get("rescale") or ("climatology" if reference_mode == "climatology" else "sample_rms")
        if rescale not in self.RESCALINGS:
            raise ValueError(f"modulation.rescale must be one of {self.RESCALINGS}, got '{rescale}'.")
        if rescale == "climatology" and reference_mode != "climatology":
            raise ValueError("modulation.rescale 'climatology' needs reference 'climatology'.")
        if reference_mode != "climatology":
            return None, None, reference_mode, rescale

        normalizer = modulation.get("normalizer")
        if normalizer is None:
            raise ValueError(
                "modulation.reference 'climatology' needs modulation.normalizer: how the data config normalises "
                "the spread variables (e.g. 'std'), to express their climatological mean in the model's units."
            )
        if statistics is None or name_to_index_stats is None:
            raise ValueError(
                "modulation.reference 'climatology' needs the dataset statistics, which the model supplies."
            )
        reference, mean_square = climatological_spread_statistics(
            spread_variables, statistics, name_to_index_stats, normalizer
        )
        return reference, (mean_square if rescale == "climatology" else None), reference_mode, rescale

    def _smoothing_kT(self, modulation: dict) -> list[Optional[float]]:
        """Smoothing ``kT`` of each scaled channel's multiplier, ``None`` for no smoothing."""
        to_channel = bool(modulation.get("smooth_to_channel", False))
        default_km = modulation.get("smoothing_km", 100.0)
        by_variable = dict(modulation.get("smoothing_km_by_variable") or {})

        smoothing = []
        for index in self.modulated_channels:
            channel = self.channels[index]
            if channel.smoothing_km is not None:
                smoothing.append(kT_from_length_scale(channel.smoothing_km))
                continue
            if to_channel:
                if channel.kT is None:
                    raise ValueError(
                        f"smooth_to_channel smooths at a channel's own kT, but '{channel.name}' has a tabulated "
                        "spectrum; give it smoothing_km."
                    )
                smoothing.append(channel.kT)
                continue
            scales = []
            for entry in channel.spread:
                short = entry[len(self.variable_prefix) :] if entry.startswith(self.variable_prefix) else entry
                value = by_variable.get(short, by_variable.get(strip_level(short), default_km))
                if value is not None:
                    scales.append(float(value))
            smoothing.append(kT_from_length_scale(max(scales)) if scales else None)
        return smoothing

    def _log_layout(
        self,
        reference_mode: str,
        normalizer: Optional[str],
        rescale: str,
        smoothing_kT: list[Optional[float]],
        variance: Tensor,
        quantile: float,
    ) -> None:
        limits = band_limit(variance, quantile)
        smoothing_km = [round(EARTH_RADIUS_KM * math.sqrt(2 * kT)) if kT is not None else None for kT in smoothing_kT]
        LOGGER.info(
            "SphericalInputConditionedNoise: %d of %d channels scaled by %d '%s' inputs (%d distinct groups); "
            "reference=%s%s, rescale=%s, clip=%s; multiplier smoothing (km: channels) %s; band limits l=%d-%d, "
            "filter truncation %d.",
            len(self.modulated_channels),
            self.n_channels,
            len(self._consumed_input_idx),
            self.variable_prefix,
            len(self.std_modulation.groups),
            reference_mode,
            f" (normalizer {normalizer})" if reference_mode == "climatology" else "",
            rescale,
            self.std_modulation.clip,
            dict(sorted(Counter(smoothing_km).items(), key=lambda item: (item[0] is None, item[0]))),
            int(limits.min()),
            int(limits.max()),
            self.spectral_filter.truncation,
        )
        if self.std_modulation.channel_scale is not None:
            LOGGER.info(
                "SphericalInputConditionedNoise: fixed rescale of the multipliers %.2f-%.2f.",
                self.std_modulation.channel_scale.min().item(),
                self.std_modulation.channel_scale.max().item(),
            )
        for row, index in enumerate(self.modulated_channels):
            channel = self.channels[index]
            LOGGER.debug(
                "  channel %d %s: %s spectrum, spread %s, smoothing %s km, band up to degree %d",
                index,
                channel.name,
                "kT" if channel.kT is not None else "tabulated",
                list(channel.spread),
                smoothing_km[row],
                int(limits[row]),
            )

    @property
    def consumed_input_idx(self) -> list[int]:
        return list(self._consumed_input_idx)

    @property
    def conditioned(self) -> bool:
        return self.enabled

    def advance(
        self,
        *,
        fcstep: int,
        batch_size: int,
        ensemble_size: int,
        member_offset: int = 0,
        num_members_total: Optional[int] = None,
        group_id: int = 0,
        device: Optional[torch.device] = None,
        inputs: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        r"""Step the noise process and, at ``fcstep == 0``, scale it by ``inputs``.

        ``inputs`` holds the :attr:`consumed_input_idx` channels on the full grid,
        shape ``(batch, time, points, variables)``. It is required at
        ``fcstep == 0`` and ignored afterwards, when the scaled field decays
        through the noise process on its own.
        """
        super().advance(
            fcstep=fcstep,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            member_offset=member_offset,
            num_members_total=num_members_total,
            group_id=group_id,
            device=device,
            seed=seed,
        )
        if not self.enabled or fcstep > 0:
            return
        if inputs is None:
            raise ValueError(
                f"SphericalInputConditionedNoise needs the '{self.variable_prefix}' inputs at the first rollout step."
            )
        self._modulate(inputs)

    @torch.no_grad()
    def _modulate(self, inputs: Tensor) -> None:
        """Replace the scaled channels of the freshly drawn state by their filtered, scaled versions.

        Works through the channels in blocks, so its transient memory does not scale
        with their number.
        """
        expected = (self.noise.state.shape[0], self.num_time_steps, self.num_grid_points, len(self._consumed_input_idx))
        if tuple(inputs.shape) != expected:
            raise ValueError(
                f"SphericalInputConditionedNoise: inputs have shape {tuple(inputs.shape)}, expected {expected} "
                "(batch, time, points, variables)."
            )

        log = not self._modulation_seen or LOGGER.isEnabledFor(logging.DEBUG)
        before_total = after_total = 0.0
        with torch.amp.autocast(device_type=inputs.device.type, enabled=False):
            amplitude = self.std_modulation(inputs)  # (batch, time, scaled channels, points)
            for channels, rows in self._modulated_blocks:
                product = self.noise.transform_channels(channels) * amplitude[:, None, :, rows]
                coeffs = self.spectral_filter.analyse(product) * self.band_response[rows].unsqueeze(-1)
                if self.preserve_filtered_variance or log:
                    before = self.std_modulation.area_mean(product**2)
                    after = self.std_modulation.area_mean(self.spectral_filter.synthesise(coeffs) ** 2)
                    if self.preserve_filtered_variance:
                        scale = torch.sqrt(before / after.clamp(min=torch.finfo(after.dtype).tiny))
                        coeffs = coeffs * scale[..., None, None]
                    if log:
                        before_total = before_total + before.sum()
                        after_total = after_total + after.sum()

                # Coefficients below the filter truncation carry over unchanged at the full lmax.
                state = self.noise.state[:, :, :, channels]  # a view: written in place
                degrees = coeffs.shape[-1]
                state.zero_()
                state[..., :degrees, :degrees, 0] = coeffs.real
                state[..., :degrees, :degrees, 1] = coeffs.imag

        if log:
            # .item() synchronises with the device, so only pay for it when the line is emitted.
            first_use = not self._modulation_seen
            self._modulation_seen = True
            LOGGER.log(
                logging.INFO if first_use else logging.DEBUG,
                "SphericalInputConditionedNoise: scaled draw; multiplier in [%.3f, %.3f] (area-mean square %.2f), "
                "%.2f%% of the scaled variance lay outside the channel bands",
                amplitude.min().item(),
                amplitude.max().item(),
                self.std_modulation.area_mean(amplitude**2).mean().item(),
                100.0 * (1.0 - after_total / before_total).item(),
            )


class BaseNoiseInjector(nn.Module, ABC):
    """Abstract base class for noise injection strategies.

    Subclasses must implement the forward method which takes an input tensor
    and returns a tuple of (modified_tensor, noise_or_none).
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass for noise injection.

        Parameters
        ----------
        x : Tensor
            Input tensor to potentially modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        grid_shard_sizes : ShardSizes
            Per-rank partition sizes along the sharded dimension, or None if not sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, Optional[Tensor]]
            Tuple of (output_tensor, noise_tensor_or_none):
                - output_tensor: The (potentially) modified input tensor
                - noise_tensor_or_none: The noise tensor for conditioning,
                  or None if noise is injected directly into output_tensor
        """
        ...


class NoOpNoiseInjector(BaseNoiseInjector):
    """No-op noise injector that passes through input unchanged.

    Use this when noise injection is disabled.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize NoOpNoiseInjector."""
        super().__init__()

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Pass through input unchanged with no noise."""
        return x, None


class NoiseConditioning(BaseNoiseInjector):
    """Noise Conditioning."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
        noise_edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        row_normalize_noise_matrix: bool = False,
        autocast: bool = False,
        sparse_projector_num_chunks: int = 1,
        num_channels: Optional[int] = None,
        graph_data: Optional[HeteroData] = None,
    ) -> None:
        """Initialize NoiseConditioning."""
        super().__init__()
        assert noise_channels_dim > 0, "Noise channels must be a positive integer"
        assert noise_mlp_hidden_dim > 0, "Noise channels must be a positive integer"

        self.noise_std = noise_std

        # Noise channels
        self.noise_channels = noise_channels_dim

        self.layer_factory = load_layer_kernels(layer_kernels)

        self.noise_mlp = MLP(
            noise_channels_dim,
            noise_mlp_hidden_dim,
            noise_channels_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=0,
            final_activation=False,
            layer_norm=True,
        )

        self.noise_graph_provider = None
        self._sparse_projector = None
        assert not (
            noise_matrix is not None and noise_edges_name is not None
        ), "Specify either noise_matrix or noise_edges_name, not both."

        if noise_edges_name is not None:
            assert graph_data is not None, "graph_data must be provided when using noise_edges_name."
            self.noise_graph_provider = ProjectionGraphProvider(
                graph=graph_data,
                edges_name=tuple(noise_edges_name),
                edge_weight_attribute=edge_weight_attribute,
                row_normalize=row_normalize_noise_matrix,
            )
            self._sparse_projector = SparseProjector(autocast=autocast, num_chunks=sparse_projector_num_chunks)
            LOGGER.info("Noise projector matrix shape = %s", self.noise_graph_provider.projection_matrix.shape)

        if noise_matrix is not None:
            self.noise_graph_provider = ProjectionGraphProvider(
                file_path=noise_matrix,
                row_normalize=row_normalize_noise_matrix,
            )
            self._sparse_projector = SparseProjector(autocast=autocast, num_chunks=sparse_projector_num_chunks)
            LOGGER.info("Noise projector matrix shape = %s", self.noise_graph_provider.projection_matrix.shape)

        LOGGER.info("processor noise channels = %d", self.noise_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:

        noise_shape = (
            batch_size,
            ensemble_size,
            grid_size if self.noise_graph_provider is None else self.noise_graph_provider.projection_matrix.shape[1],
            self.noise_channels,
        )

        noise = torch.randn(size=noise_shape, dtype=noise_dtype, device=x.device) * self.noise_std
        noise.requires_grad = False

        if self.noise_graph_provider is not None:
            channel_shard_sizes = get_shard_sizes(noise, -1, model_comm_group)
            noise = shard_tensor(noise, -1, channel_shard_sizes, model_comm_group)  # split across channels

            noise = einops.rearrange(
                noise, "batch ensemble grid vars -> (batch ensemble) grid vars"
            )  # batch and ensemble always 1 when sharded

            projection_matrix = self.noise_graph_provider.get_edges(device=noise.device)
            noise = self._sparse_projector(noise, projection_matrix)  # to shape of hidden grid

            noise = einops.rearrange(noise, "bse grid vars -> (bse grid) vars")  # shape of x
            noise = all_to_all_transpose(
                noise, 0, grid_shard_sizes, -1, channel_shard_sizes, model_comm_group
            )  # sharded grid dim, full channels
        else:
            noise = einops.rearrange(noise, "batch ensemble grid vars -> (batch ensemble grid) vars")  # shape of x
            noise_shard_sizes = get_shard_sizes(noise, 0, model_comm_group)
            noise = shard_tensor(noise, 0, noise_shard_sizes, model_comm_group)  # sharded grid dim, full channels

        noise = checkpoint(self.noise_mlp, noise, use_reentrant=False)

        LOGGER.debug("Noise noise.shape = %s, noise.norm: %.9e", noise.shape, torch.linalg.norm(noise))

        return x, noise


class NoiseInjector(BaseNoiseInjector):
    """Noise Injection Module.

    Generates noise and projects it directly into the input tensor,
    returning None for the noise (since it's already incorporated).
    """

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        num_channels: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
        graph_data: Optional[HeteroData] = None,
    ) -> None:
        """Initialize NoiseInjector.

        Parameters
        ----------
        noise_std : int
            Standard deviation for noise generation
        noise_channels_dim : int
            Number of noise channels
        noise_mlp_hidden_dim : int
            Hidden dimension of noise MLP
        num_channels : int
            Number of model channels for projection
        layer_kernels : DotDict
            Layer kernel configurations
        noise_matrix : str, optional
            Optional path to noise truncation matrix
        graph_data : Optional[HeteroData], optional
            Graph data for noise conditioning.
        """
        super().__init__()

        self._noise_conditioning = NoiseConditioning(
            noise_std=noise_std,
            noise_channels_dim=noise_channels_dim,
            noise_mlp_hidden_dim=noise_mlp_hidden_dim,
            layer_kernels=layer_kernels,
            noise_matrix=noise_matrix,
            graph_data=graph_data,
        )
        self.noise_channels = noise_channels_dim
        self.projection = nn.Linear(num_channels + self.noise_channels, num_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Generate noise and inject it into the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor to modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        grid_shard_sizes : ShardSizes
            Per-rank partition sizes along the sharded dimension, or None if not sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, None]
            Tuple of (modified_x, None): Modified tensor with noise injected
        """
        x, noise = self._noise_conditioning(
            x=x,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=grid_size,
            grid_shard_sizes=grid_shard_sizes,
            noise_dtype=noise_dtype,
            model_comm_group=model_comm_group,
        )

        return (
            self.projection(torch.cat([x, noise], dim=-1)),
            None,
        )

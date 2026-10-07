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
from anemoi.models.layers.noise_modulation import resolve_prefixed_variables
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter
from anemoi.models.layers.spherical_noise import DEFAULT_DIFFUSION_KT
from anemoi.models.layers.spherical_noise import EARTH_RADIUS_KM
from anemoi.models.layers.spherical_noise import BaseSphericalNoise
from anemoi.models.layers.spherical_noise import build_inverse_sht
from anemoi.models.layers.spherical_noise import build_noise
from anemoi.models.layers.spherical_noise import diffusion_band_limit
from anemoi.models.layers.spherical_noise import diffusion_lowpass_response
from anemoi.models.layers.spherical_noise import filter_truncation
from anemoi.models.layers.spherical_noise import heat_kernel_response
from anemoi.models.layers.spherical_noise import noise_seeds_reflects
from anemoi.models.layers.spherical_noise import per_channel_values
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


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
        eight, each with a different spatial correlation length.
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
    """

    def __init__(
        self,
        *,
        grid: Union[str, int],
        noise: dict,
        n_channels: int = 1,
        centered: bool = False,
        dataset: Optional[str] = None,
        default_lambd: float = 1.0,
        num_time_steps: int = 1,
        num_grid_points: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Hydra hands over OmegaConf containers; resolve to plain Python once here so
        # nothing downstream has to cope with ListConfig.
        if OmegaConf.is_config(noise):
            noise = OmegaConf.to_container(noise, resolve=True)
        self.noise_params = dict(noise)
        self.n_channels = n_channels
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
        """
        layout = (ensemble_size, member_offset, num_members_total, group_id)
        if self.noise is None or layout != self._layout:
            seeds, reflects = noise_seeds_reflects(
                ensemble_size,
                centered=self.centered,
                member_offset=member_offset,
                num_members_total=num_members_total,
                group_id=group_id,
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
            ).to(device)
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
    r"""FourCastNet 3 input perturbation whose amplitude follows the analysis spread.

    :class:`SphericalInputNoise` injects noise of the same amplitude everywhere.
    This variant keeps its channels -- the same ``kT`` scale ladder, the same
    Ornstein-Uhlenbeck process in time -- but redistributes each channel in space
    so it is strong where the initial conditions are uncertain and weak where
    they are not, using ensemble spread fields (e.g. the ERA5 EDA standard
    deviation) read from the model input.

    At the start of every rollout (``fcstep == 0``):

    1. :class:`~anemoi.models.layers.noise_modulation.GroupedStdModulation` turns
       the spread fields of each history step into one amplitude map
       :math:`M_c` per channel, from the variable group assigned to it,
       smoothed to the channel's own correlation scale.
    2. The freshly drawn stationary noise :math:`\eta_c` is multiplied by
       :math:`M_c`.
    3. Multiplying in grid space convolves the two spectra, which leaks energy
       out of the channel's scale band -- mostly to smaller scales, increasingly
       so for the large-scale channels. Each product is therefore low-pass
       filtered back to its own band (:func:`diffusion_lowpass_response`), so a
       channel keeps representing the scale it was designed for.
    4. With ``preserve_total_variance`` the filtered field is rescaled to the
       area-weighted variance it had before filtering: the filter moves energy
       back into the band rather than discarding it.
    5. The result is written back into the spectral state of the noise process.

    Later rollout steps need no spread fields: the unchanged FourCastNet 3 update
    :math:`\eta \leftarrow \phi\,\eta + \sqrt{1-\phi^2}\,\sigma_l\,\xi` carries the
    modulated field forward, so the initial-condition structure decays by
    :math:`\phi` per step while fresh homogeneous noise takes over.

    The consumed spread channels are dropped from the encoder input by the model,
    so they shape the noise without becoming input features. With
    ``modulation.enabled: False`` they are still dropped and the noise is exactly
    that of :class:`SphericalInputNoise` -- a matched baseline from the same data.

    Parameters
    ----------
    modulation : dict
        ``enabled``; ``groups`` (group name to variables); ``channel_group`` (group
        of each channel); ``variable_prefix`` (``"std_"``); ``source``
        (``"eda_stdev"``); ``area_weight``; ``smooth_to_channel``; ``clip``;
        ``preserve_total_variance``; ``band_filter`` (``quantile``, ``taper``).
    name_to_index : dict
        Supplied by the model: the dataset's model-input name to position map.
    **kwargs
        Passed to :class:`SphericalInputNoise`.
    """

    def __init__(self, *, modulation: dict, name_to_index: Optional[dict[str, int]] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if name_to_index is None:
            raise ValueError("SphericalInputConditionedNoise needs name_to_index, which the model supplies.")
        if OmegaConf.is_config(modulation):
            modulation = OmegaConf.to_container(modulation, resolve=True)
        modulation = dict(modulation)

        self.enabled = bool(modulation.get("enabled", True))
        self.variable_prefix = modulation.get("variable_prefix", "std_")
        self.preserve_total_variance = bool(modulation.get("preserve_total_variance", True))
        source = modulation.get("source", "eda_stdev")
        if source != "eda_stdev":
            raise ValueError(f"modulation.source '{source}' is not supported; expected 'eda_stdev'.")

        self._consumed_input_idx = [
            index for index, _ in resolve_prefixed_variables(name_to_index, self.variable_prefix)
        ]
        self.std_modulation: Optional[GroupedStdModulation] = None
        self.spectral_filter: Optional[SphericalSpectralFilter] = None
        self._modulation_seen = False

        if not self.enabled:
            LOGGER.info(
                "SphericalInputConditionedNoise: modulation disabled; %d '%s' inputs are dropped from the "
                "encoder input and the noise is plain FourCastNet 3.",
                len(self._consumed_input_idx),
                self.variable_prefix,
            )
            return

        if self.noise_params.get("type") != "diffusion":
            raise ValueError(
                "SphericalInputConditionedNoise modulates the spectral state of a 'diffusion' noise field, got "
                f"type '{self.noise_params.get('type')}'."
            )
        channel_group = list(modulation.get("channel_group", []))
        if len(channel_group) != self.n_channels:
            raise ValueError(
                f"modulation.channel_group has {len(channel_group)} entries but there are {self.n_channels} channels."
            )

        band = dict(modulation.get("band_filter") or {})
        kT = per_channel_values(self.noise_params.get("kT", DEFAULT_DIFFUSION_KT), self.n_channels, "kT")
        band_response = diffusion_lowpass_response(
            kT, self.lmax, quantile=float(band.get("quantile", 0.99)), taper=float(band.get("taper", 0.25))
        )
        smooth = bool(modulation.get("smooth_to_channel", True))
        smooth_response = heat_kernel_response(kT, self.lmax) if smooth else None

        # The filter only needs the degrees its responses reach, far fewer than lmax.
        truncation = filter_truncation([r for r in (band_response, smooth_response) if r is not None], self.lmax)
        lons_per_lat = self._transform[0].lons_per_lat
        self.spectral_filter = SphericalSpectralFilter(lons_per_lat, truncation)
        self.register_buffer("band_response", band_response[:, : truncation + 1].to(torch.float32), persistent=False)

        if modulation.get("area_weight", True):
            area_weights = torch.as_tensor(quadrature_weights(lons_per_lat))
        else:
            area_weights = torch.full((self.num_grid_points,), 1.0 / self.num_grid_points)

        self.std_modulation = GroupedStdModulation(
            groups=modulation.get("groups", {}),
            channel_group=channel_group,
            name_to_index=name_to_index,
            area_weights=area_weights,
            variable_prefix=self.variable_prefix,
            clip=modulation.get("clip", (0.25, 4.0)),
            preserve_total_variance=self.preserve_total_variance,
            spectral_filter=self.spectral_filter if smooth else None,
            smooth_response=smooth_response[:, : truncation + 1] if smooth else None,
        )

        band_limit = diffusion_band_limit(kT, self.lmax, float(band.get("quantile", 0.99)))
        LOGGER.info(
            "SphericalInputConditionedNoise: %d '%s' inputs, group sizes %s, filter truncation %d, "
            "smooth_to_channel=%s, area_weight=%s, clip=%s, preserve_total_variance=%s",
            len(self._consumed_input_idx),
            self.variable_prefix,
            self.std_modulation.group_sizes(),
            truncation,
            smooth,
            modulation.get("area_weight", True),
            self.std_modulation.clip,
            self.preserve_total_variance,
        )
        for channel, (group, kT_c, limit) in enumerate(zip(channel_group, kT.tolist(), band_limit.tolist())):
            LOGGER.info(
                "  channel %d: group %s, kT=%.4e (L=%.0f km), band up to degree %d",
                channel,
                group,
                kT_c,
                EARTH_RADIUS_KM * math.sqrt(2 * kT_c),
                limit,
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
    ) -> None:
        r"""Step the noise process and, at ``fcstep == 0``, modulate it by ``inputs``.

        ``inputs`` holds the :attr:`consumed_input_idx` channels on the full grid,
        shape ``(batch, time, points, variables)``. It is required at
        ``fcstep == 0`` and ignored afterwards, when the modulated field decays
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
        """Replace the freshly drawn noise state by its filtered, modulated version."""
        expected = (self.noise.state.shape[0], self.num_time_steps, self.num_grid_points, len(self._consumed_input_idx))
        if tuple(inputs.shape) != expected:
            raise ValueError(
                f"SphericalInputConditionedNoise: inputs have shape {tuple(inputs.shape)}, expected {expected} "
                "(batch, time, points, variables)."
            )

        with torch.amp.autocast(device_type=inputs.device.type, enabled=False):
            amplitude = self.std_modulation(inputs)  # (batch, time, channels, points)
            product = self.noise() * amplitude.unsqueeze(1)  # (batch, ensemble, time, channels, points)

            coeffs = self.spectral_filter.analyse(product) * self.band_response.unsqueeze(-1)
            filtered = self.spectral_filter.synthesise(coeffs)
            before = self.std_modulation.area_mean(product**2)
            after = self.std_modulation.area_mean(filtered**2)
            if self.preserve_total_variance:
                coeffs = coeffs * torch.sqrt(before / after.clamp(min=torch.finfo(after.dtype).tiny))[..., None, None]

        # Coefficients below the filter truncation carry over unchanged at the full lmax.
        state = torch.zeros_like(self.noise.state)
        degrees = coeffs.shape[-1]
        state[..., :degrees, :degrees, 0] = coeffs.real
        state[..., :degrees, :degrees, 1] = coeffs.imag
        self.noise.set_tensor_state(state)

        # .item() synchronises with the device, so only pay for it when the line is emitted.
        first_use = not self._modulation_seen
        if first_use or LOGGER.isEnabledFor(logging.DEBUG):
            self._modulation_seen = True
            LOGGER.log(
                logging.INFO if first_use else logging.DEBUG,
                "SphericalInputConditionedNoise: modulated draw; amplitude in [%.3f, %.3f], "
                "%.2f%% of the modulated variance lay outside the channel bands",
                amplitude.min().item(),
                amplitude.max().item(),
                100.0 * (1.0 - after.sum() / before.sum()).item(),
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

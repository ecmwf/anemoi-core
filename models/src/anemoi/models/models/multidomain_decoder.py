# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

"""Paper-compatible components for latent-to-native residual decoding.

This module deliberately contains the decoder pieces whose behaviour is part
of the scientific contract rather than of a particular Anemoi graph object:

* the zero-initialised native-grid convolutional correction; and
* the temperature-coupled saturation coordinate used for specific humidity.

The graph-transformer backward mapper remains responsible for moving the
100,000-node factor state to a regional grid.  These components are applied
after that mapper, exactly as in the recovered paper implementation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class HumidityTemperaturePair:
    """One pressure-level humidity/temperature pair in the union catalogue."""

    level_hpa: int
    humidity_union_index: int
    temperature_union_index: int


def humidity_temperature_pairs(
    channel_metadata: str | Path,
    domain: str,
) -> tuple[HumidityTemperaturePair, ...]:
    """Return the physically valid q/T pairs available in one domain.

    A humidity field is included only if temperature at the identical pressure
    is also present.  This matters for AROME Arctic, whose catalogue contains
    q at 150 hPa but no co-located temperature, and for CERRA, which contains
    no pressure-level specific humidity.
    """

    with Path(channel_metadata).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or domain not in rows[0]:
        raise KeyError(f"Domain {domain!r} is absent from {channel_metadata}")
    pressure_rows: dict[tuple[str, str], Mapping[str, str]] = {
        (row["variable"], row["pressure_pa"]): row
        for row in rows
        if row.get("level_type") == "pressure" and row.get("pressure_pa")
    }
    pairs = []
    for (variable, pressure_pa), humidity in pressure_rows.items():
        if variable != "q" or humidity.get(domain, "False").lower() != "true":
            continue
        temperature = pressure_rows.get(("t", pressure_pa))
        if temperature is None or temperature.get(domain, "False").lower() != "true":
            continue
        pairs.append(
            HumidityTemperaturePair(
                level_hpa=int(round(float(pressure_pa) / 100.0)),
                humidity_union_index=int(humidity["union_index"]),
                temperature_union_index=int(temperature["union_index"]),
            )
        )
    return tuple(sorted(pairs, key=lambda pair: pair.level_hpa))


def saturation_specific_humidity(
    temperature_k: torch.Tensor,
    pressure_hpa: torch.Tensor,
) -> torch.Tensor:
    """Mixed-phase Buck saturation specific humidity in kg kg-1.

    This is the recovered paper formula: Buck water and ice vapour pressures
    blended linearly between 250.15 K and 273.15 K.
    """

    temperature_k = torch.clamp(temperature_k.float(), min=150.0, max=350.0)
    tc = temperature_k - 273.15
    es_water = 611.21 * torch.exp((18.678 - tc / 234.5) * (tc / (257.14 + tc)))
    es_ice = 611.15 * torch.exp((23.036 - tc / 333.7) * (tc / (279.82 + tc)))
    water_weight = torch.clamp((temperature_k - 250.15) / 23.0, min=0.0, max=1.0)
    vapour_pressure = water_weight * es_water + (1.0 - water_weight) * es_ice
    pressure_pa = pressure_hpa.float() * 100.0
    vapour_pressure = torch.minimum(
        torch.clamp(vapour_pressure, min=1.0e-6),
        0.95 * pressure_pa,
    )
    return 0.622 * vapour_pressure / (pressure_pa - 0.378 * vapour_pressure)


class SaturationHumidityCoordinate(torch.nn.Module):
    """Pressure-level q coordinate coupled to generated temperature.

    ``bounded_logit`` reproduces the coordinate stated in the paper.  The
    later ``softplus_ratio`` continuation is supported because its recovered
    checkpoint removes the upper-bound pile-up while retaining the same
    temperature coupling.  Caps/scales and q physical scales must be fitted
    using training cases only.
    """

    def __init__(
        self,
        levels_hpa: Sequence[int],
        ratio_scales: Sequence[float],
        q_physical_scales: Sequence[float],
        *,
        epsilon: float = 1.0e-5,
        coordinate_kind: str = "bounded_logit",
    ) -> None:
        super().__init__()
        if coordinate_kind not in {"bounded_logit", "softplus_ratio"}:
            raise ValueError(f"Unsupported humidity coordinate {coordinate_kind!r}")
        if not (len(levels_hpa) == len(ratio_scales) == len(q_physical_scales) and len(levels_hpa) > 0):
            raise ValueError("Humidity levels, ratio scales and q scales must align")
        if epsilon <= 0.0 or epsilon >= 0.5:
            raise ValueError("Humidity epsilon must lie in (0, 0.5)")
        levels = torch.as_tensor(levels_hpa, dtype=torch.float32)
        ratios = torch.as_tensor(ratio_scales, dtype=torch.float32)
        q_scales = torch.as_tensor(q_physical_scales, dtype=torch.float32)
        if torch.any(ratios <= 0.0) or torch.any(q_scales <= 0.0):
            raise ValueError("Humidity scales must be positive")
        self.coordinate_kind = coordinate_kind
        self.epsilon = float(epsilon)
        self.register_buffer("levels_hpa", levels)
        self.register_buffer("ratio_scales", ratios)
        self.register_buffer("q_physical_scales", q_scales)

    def _shape(self, values: torch.Tensor) -> tuple[int, ...]:
        if values.shape[-1] != len(self.levels_hpa):
            raise ValueError(f"Humidity tensor has {values.shape[-1]} channels; expected {len(self.levels_hpa)}")
        return (1,) * (values.ndim - 1) + (len(self.levels_hpa),)

    def coordinate(self, q: torch.Tensor, temperature_k: torch.Tensor) -> torch.Tensor:
        if q.shape != temperature_k.shape:
            raise ValueError("Humidity and temperature tensors must have identical shapes")
        shape = self._shape(q)
        qsat = saturation_specific_humidity(
            temperature_k,
            self.levels_hpa.to(q.device).view(shape),
        )
        ratio = torch.clamp(q.float(), min=0.0) / (self.ratio_scales.to(q.device).view(shape) * qsat)
        if self.coordinate_kind == "softplus_ratio":
            ratio = torch.clamp(ratio, min=self.epsilon)
            return ratio + torch.log(-torch.expm1(-ratio))
        fraction = torch.clamp(ratio, min=self.epsilon, max=1.0 - self.epsilon)
        return torch.logit(fraction)

    def inverse(self, coordinate: torch.Tensor, temperature_k: torch.Tensor) -> torch.Tensor:
        if coordinate.shape != temperature_k.shape:
            raise ValueError("Humidity coordinate and temperature tensors must align")
        shape = self._shape(coordinate)
        ratio = (
            F.softplus(coordinate.float())
            if self.coordinate_kind == "softplus_ratio"
            else torch.sigmoid(coordinate.float())
        )
        qsat = saturation_specific_humidity(
            temperature_k,
            self.levels_hpa.to(coordinate.device).view(shape),
        )
        return self.ratio_scales.to(coordinate.device).view(shape) * qsat * ratio

    def physical_mse(
        self,
        coordinate: torch.Tensor,
        generated_temperature_k: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.inverse(coordinate, generated_temperature_k)
        shape = self._shape(prediction)
        scales = self.q_physical_scales.to(prediction.device).view(shape)
        return ((prediction - target_q.float()) / scales).square().mean()


class LocalGridResidualBlock(torch.nn.Module):
    """The paper decoder's depthwise native-grid residual block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            padding_mode="replicate",
        )
        self.pointwise = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(F.gelu(self.depthwise(values)))
        return values + update


class LocalGridRefiner(torch.nn.Module):
    """Zero-initialised additive correction on a regular regional grid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        width: int = 96,
        dilations: Iterable[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        dilations = tuple(int(value) for value in dilations)
        if width < 1 or not dilations or any(value < 1 for value in dilations):
            raise ValueError("Local refiner width and dilations must be positive")
        self.input_projection = torch.nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = torch.nn.Sequential(*(LocalGridResidualBlock(width, dilation) for dilation in dilations))
        self.output_projection = torch.nn.Conv2d(width, out_channels, kernel_size=1)
        torch.nn.init.zeros_(self.output_projection.weight)
        torch.nn.init.zeros_(self.output_projection.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.gelu(self.input_projection(values))
        return self.output_projection(self.blocks(values))


def apply_context_module(
    module: torch.nn.Module,
    *values: torch.Tensor,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply a native head without retaining its large concatenated activations.

    Concatenation must occur inside the non-reentrant checkpoint: storing the
    concatenated input alone costs 8.6 GiB on the full UWC-West grid. This also
    covers gradients through frozen atmospheric refinement during the later
    precipitation stage, without altering its full-domain computation.
    """

    def apply(*inputs: torch.Tensor) -> torch.Tensor:
        joined = torch.cat(inputs, dim=-1)
        if grid_shape is None:
            return module(joined)
        height, width = grid_shape
        image = joined.T.reshape(1, joined.shape[-1], height, width)
        result = module(image)
        return result.reshape(result.shape[1], height * width).T

    if module.training and torch.is_grad_enabled():
        return checkpoint(apply, *values, use_reentrant=False)
    return apply(*values)


def _safe_key(value: str) -> str:
    return value.lower().replace("-", "_")


class PaperMultidomainDecoder(torch.nn.Module):
    """Paper-structured factor decoder shared over regional products.

    The expensive learned path is one graph-transformer backward mapper, not a
    second latent processor.  All semantic fields live in a fixed union so the
    mapper and native-grid refiner are shared; unavailable fields are marked by
    an explicit validity vector and are removed only after decoding.
    """

    def __init__(
        self,
        *,
        factors: int,
        static_features: int,
        union_channels: int,
        domain_native_points: Mapping[str, int],
        pools: int,
        humidity_union_indices: Sequence[int],
        width: int = 256,
        heads: int = 8,
        chunks: int = 4,
        learned_edge_channels: int = 8,
        local_width: int = 96,
        local_dilations: Sequence[int] = (1, 2, 4, 8),
        precipitation_width: int = 32,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Decoder width must be divisible by its attention heads")
        if learned_edge_channels < 1:
            raise ValueError("The paper decoder requires learned edge channels")
        try:
            from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
        except ImportError as exc:  # pragma: no cover - exercised in the training environment
            raise ImportError("PaperMultidomainDecoder requires the Anemoi model package") from exc

        self.factors = int(factors)
        self.static_features = int(static_features)
        self.union_channels = int(union_channels)
        self.width = int(width)
        self.learned_edge_channels = int(learned_edge_channels)
        self.domain_names = tuple(domain_native_points)
        self.domain_to_index = {name: index for index, name in enumerate(self.domain_names)}
        self.pool_embedding = torch.nn.Embedding(pools, 24)
        self.domain_embedding = torch.nn.Embedding(len(self.domain_names), 24)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(6, 48),
            torch.nn.SiLU(),
            torch.nn.Linear(48, 48),
        )
        self.condition_channels = 96
        self.factor_lift = torch.nn.Sequential(
            torch.nn.Linear(self.factors + self.static_features, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        # Normalised deterministic union fields, their validity mask, physical
        # high-resolution statics and domain/cadence/lead context.
        self.destination_context_channels = 2 * self.union_channels + self.static_features + self.condition_channels
        offsets = {}
        offset = 0
        for domain, points in domain_native_points.items():
            offsets[domain] = offset
            offset += int(points)
        self.domain_edge_offsets = offsets
        # One parameter tensor keeps the DDP graph identical even when ranks
        # train different domains. Domain offsets preserve independent learned
        # destination-node channels, matching the paper provider semantics.
        self.edge_embedding = torch.nn.Embedding(offset, learned_edge_channels)
        self.mapper = GraphTransformerBackwardMapper(
            in_channels_src=width,
            in_channels_dst=self.destination_context_channels,
            num_channels=width,
            out_channels_dst=self.union_channels,
            num_chunks=chunks,
            num_heads=heads,
            mlp_hidden_ratio=4.0,
            edge_dim=3 + learned_edge_channels,
            qk_norm=False,
            mlp_implementation="mlp",
            # The recovered base decoder starts as the fixed PCA skip.  Its
            # learned mesh-to-grid extraction is introduced from zero.
            initialise_data_extractor_zero=True,
            cpu_offload=False,
            gradient_checkpointing=True,
            layer_kernels=None,
            shard_strategy="edges",
            graph_attention_backend="triton",
            edge_pre_mlp=False,
        )
        local_input_channels = self.union_channels + self.destination_context_channels + self.union_channels
        self.local_grid_refiner = LocalGridRefiner(
            local_input_channels,
            self.union_channels,
            width=local_width,
            dilations=local_dilations,
        )
        self.precipitation_lift = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        torch.nn.init.zeros_(self.precipitation_lift[-1].weight)
        torch.nn.init.zeros_(self.precipitation_lift[-1].bias)
        special_input_channels = self.union_channels + self.destination_context_channels
        self.precipitation_head = torch.nn.Sequential(
            torch.nn.LayerNorm(special_input_channels),
            torch.nn.Linear(special_input_channels, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, 1),
        )
        torch.nn.init.zeros_(self.precipitation_head[-1].weight)
        torch.nn.init.zeros_(self.precipitation_head[-1].bias)
        self.local_grid_precipitation_refiner = LocalGridRefiner(
            1 + special_input_channels + self.union_channels,
            1,
            width=precipitation_width,
            dilations=local_dilations,
        )
        humidity_indices = torch.as_tensor(humidity_union_indices, dtype=torch.long)
        if len(humidity_indices) != len(torch.unique(humidity_indices)):
            raise ValueError("Humidity union indices contain duplicates")
        self.register_buffer("humidity_union_indices", humidity_indices, persistent=True)
        self.humidity_saturation_head = torch.nn.Sequential(
            torch.nn.LayerNorm(special_input_channels),
            torch.nn.Linear(special_input_channels, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, len(humidity_indices)),
        )
        torch.nn.init.zeros_(self.humidity_saturation_head[-1].weight)
        torch.nn.init.zeros_(self.humidity_saturation_head[-1].bias)

    def condition(
        self,
        domain: str,
        pool_index: int,
        lead_hours: float,
        cadence_hours: float,
    ) -> torch.Tensor:
        device = self.pool_embedding.weight.device
        dtype = self.pool_embedding.weight.dtype
        lead = torch.tensor(float(lead_hours) / 6.0, device=device, dtype=dtype)
        cadence = torch.tensor(float(cadence_hours) / 6.0, device=device, dtype=dtype)
        phase = math.pi * lead
        physical = torch.stack(
            (lead, cadence, torch.sin(phase), torch.cos(phase), lead * cadence, lead.square())
        ).unsqueeze(0)
        domain_index = torch.tensor(self.domain_to_index[domain], device=device)
        pool = torch.tensor(pool_index, device=device)
        return torch.cat(
            (
                self.domain_embedding(domain_index),
                self.pool_embedding(pool),
                self.time_embedding(physical).squeeze(0),
            )
        )

    def forward(
        self,
        *,
        factor_mesh: torch.Tensor,
        context_native_union: torch.Tensor,
        availability_union: torch.Tensor,
        mesh_static: torch.Tensor,
        native_static: torch.Tensor,
        linear_skip_union: torch.Tensor,
        precipitation_mesh: torch.Tensor,
        precipitation_direct_skip: torch.Tensor,
        edge_index: torch.Tensor,
        physical_edge_attributes: torch.Tensor,
        domain: str,
        pool_index: int,
        lead_hours: float,
        cadence_hours: float,
        grid_shape: tuple[int, ...],
        humidity_direct_skip: torch.Tensor | None = None,
        decoder_stage: str = "joint",
    ) -> dict[str, torch.Tensor]:
        from anemoi.models.distributed.shapes import BipartiteGraphShardInfo

        nodes = context_native_union.shape[0]
        if factor_mesh.shape != (mesh_static.shape[0], self.factors):
            raise ValueError("Factor mesh and static mesh do not match the decoder contract")
        if context_native_union.shape != (nodes, self.union_channels):
            raise ValueError("Native deterministic context must use the complete semantic union")
        if linear_skip_union.shape != context_native_union.shape:
            raise ValueError("Linear PCA skip must use the complete semantic union")
        if native_static.shape != (nodes, self.static_features):
            raise ValueError("Native static context has an unexpected shape")
        availability_union = availability_union.to(context_native_union.dtype)
        if availability_union.ndim == 1:
            availability_union = availability_union.unsqueeze(0).expand(nodes, -1)
        if availability_union.shape != context_native_union.shape:
            raise ValueError("Union validity mask does not match the native context")
        if edge_index.shape[0] != 2 or physical_edge_attributes.shape != (
            edge_index.shape[1],
            3,
        ):
            raise ValueError("Decoder edges require three physical attributes")
        if int(edge_index[1].max()) >= nodes:
            raise ValueError("Decoder edge destination is outside the native domain")

        if decoder_stage not in {"base", "local", "precipitation", "humidity", "joint"}:
            raise ValueError(f"Unknown decoder stage {decoder_stage!r}")
        use_local_refiner = decoder_stage in {"local", "precipitation", "humidity", "joint"}
        use_precipitation = decoder_stage in {"precipitation", "joint"}
        use_humidity = decoder_stage in {"humidity", "joint"}

        mesh_hidden = self.factor_lift(torch.cat((factor_mesh, mesh_static), dim=-1))
        atmospheric_mesh_hidden = mesh_hidden
        if precipitation_mesh.shape != (factor_mesh.shape[0], 1):
            raise ValueError("Precipitation mesh must contain one transformed coordinate")
        if use_precipitation:
            mesh_hidden = mesh_hidden + self.precipitation_lift(precipitation_mesh)
        condition = self.condition(domain, pool_index, lead_hours, cadence_hours)
        condition = condition.to(context_native_union.dtype).unsqueeze(0).expand(nodes, -1)
        destination_context = torch.cat((context_native_union, availability_union, native_static, condition), dim=-1)
        destination_ids = edge_index[1]
        learned_edges = self.edge_embedding(destination_ids + self.domain_edge_offsets[domain])
        edge_attributes = torch.cat((physical_edge_attributes.to(learned_edges.dtype), learned_edges), dim=-1)
        decoded = self.mapper(
            (mesh_hidden, destination_context),
            batch_size=1,
            shard_info=BipartiteGraphShardInfo(),
            edge_attr=edge_attributes,
            edge_index=edge_index,
            edges_are_dst_sorted=True,
        )
        precipitation = precipitation_direct_skip
        if use_precipitation:
            precipitation = precipitation + apply_context_module(self.precipitation_head, decoded, destination_context)

        if len(grid_shape) == 2 and use_local_refiner:
            ny, nx = grid_shape
            if ny * nx != nodes:
                raise ValueError("Native grid shape does not match its node count")

            decoded = decoded + apply_context_module(
                self.local_grid_refiner,
                decoded,
                destination_context,
                linear_skip_union,
                grid_shape=(ny, nx),
            )
            # The paper's separate precipitation refiner sees the already
            # refined atmospheric correction, not the pre-refinement mapper
            # output.
            if use_precipitation:
                precipitation = precipitation + apply_context_module(
                    self.local_grid_precipitation_refiner,
                    precipitation,
                    destination_context,
                    decoded,
                    linear_skip_union,
                    grid_shape=(ny, nx),
                )
        # Rain specialization trains its latent lift while the atmospheric
        # mapper/refiner weights are frozen. That changes the mapper's input,
        # so its rain-conditioned atmospheric carrier is NOT the frozen
        # atmospheric prediction. Keep the trained rain calculation above
        # unchanged, but publish the independent atmospheric route in joint
        # inference. The partial precipitation training path stays identical
        # (one mapper call), retaining existing checkpoints and gradients.
        atmospheric_decoded = decoded
        if decoder_stage == "joint":
            atmospheric_decoded = self.mapper(
                (atmospheric_mesh_hidden, destination_context),
                batch_size=1,
                shard_info=BipartiteGraphShardInfo(),
                edge_attr=edge_attributes,
                edge_index=edge_index,
                edges_are_dst_sorted=True,
            )
            if len(grid_shape) == 2:
                atmospheric_decoded = atmospheric_decoded + apply_context_module(
                    self.local_grid_refiner,
                    atmospheric_decoded,
                    destination_context,
                    linear_skip_union,
                    grid_shape=grid_shape,
                )
        # Humidity was the final isolated specialization in the recovered
        # chain.  It consumes the final atmospheric correction, after the
        # optional native-grid refiner.
        if use_humidity:
            humidity = apply_context_module(self.humidity_saturation_head, atmospheric_decoded, destination_context)
            if humidity_direct_skip is not None:
                if humidity_direct_skip.shape != humidity.shape:
                    raise ValueError("Humidity direct skip and union humidity head differ")
                humidity = humidity + humidity_direct_skip
        else:
            humidity = decoded.new_empty((0, len(self.humidity_union_indices)))
        # Domain-specific stages may have no applicable loss on a rank (e.g.
        # TITAN has no raster refiner). Keep their zero-gradient participation
        # inside forward so DDP sees these parameters before unused detection.
        distributed_zero = decoded.new_zeros(())
        if self.training and torch.is_grad_enabled():
            for parameter in self.parameters():
                if parameter.requires_grad:
                    distributed_zero = distributed_zero + parameter.reshape(-1)[0] * 0.0
        return {
            "atmospheric_residual": linear_skip_union + atmospheric_decoded,
            "precipitation_coordinate": precipitation,
            "humidity_saturation_coordinate": humidity,
            "distributed_zero": distributed_zero,
        }

    def transfer_paper_weights(self, source_state: Mapping[str, torch.Tensor]) -> dict:
        """Transfer only shape- and meaning-compatible paper decoder weights."""

        target = self.state_dict()
        copied: list[str] = []
        skipped: list[str] = []

        def copy(source_name: str, target_name: str) -> None:
            if source_name not in source_state or target_name not in target:
                skipped.append(f"{source_name} -> {target_name}: absent")
                return
            if source_state[source_name].shape != target[target_name].shape:
                skipped.append(
                    f"{source_name} -> {target_name}: "
                    f"{tuple(source_state[source_name].shape)} != {tuple(target[target_name].shape)}"
                )
                return
            target[target_name].copy_(source_state[source_name])
            copied.append(target_name)

        # The second lift layer is independent of factor-basis dimensionality.
        for suffix in ("weight", "bias"):
            copy(f"factor_lift.2.{suffix}", f"factor_lift.2.{suffix}")
        copy("factor_lift.0.bias", "factor_lift.0.bias")
        # Graph-transformer attention and MLP processor widths/edge dimensions
        # are identical. Destination embedding and output extraction depend on
        # the multidomain semantic union and are deliberately reinitialised.
        for name in tuple(source_state):
            prefix = "decoder.meps.proc."
            if name.startswith(prefix):
                copy(name, "mapper.proc." + name[len(prefix) :])
        # The internal width-96 convolution blocks are catalogue-independent;
        # only their input and output projections depend on channel layout.
        for name in tuple(source_state):
            prefix = "local_grid_refiner.blocks."
            if name.startswith(prefix):
                copy(name, name)
        self.load_state_dict(target)
        return {
            "copied_parameter_tensors": len(copied),
            "copied": copied,
            "skipped": skipped,
            "reinitialised": [
                "factor_lift.0.weight",
                "mapper.emb_nodes_dst.*",
                "mapper.node_data_extractor.*",
                "local_grid_refiner.input_projection.*",
                "local_grid_refiner.output_projection.*",
                "precipitation_*",
                "humidity_saturation_head.*",
                "edge_embedding.*",
            ],
        }

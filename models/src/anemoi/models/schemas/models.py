# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from anemoi.models.layers.target_features import VALID_TARGET_FEATURES
from anemoi.models.schemas.schema_utils import DatasetDict
from anemoi.utils.schemas import BaseModel

from .aggregator import AggregatorSchema  # noqa: TC001
from .bounding import BoundingSchema
from .decoder import GNNDecoderSchema  # noqa: TC001
from .decoder import GraphTransformerDecoderSchema  # noqa: TC001
from .decoder import PointWiseBackwardMapperSchema  # noqa: TC001
from .decoder import TransformerDecoderSchema  # noqa: TC001
from .encoder import GNNEncoderSchema  # noqa: TC001
from .encoder import GraphTransformerEncoderSchema  # noqa: TC001
from .encoder import PointWiseForwardMapperSchema  # noqa: TC001
from .encoder import TransformerEncoderSchema  # noqa: TC001
from .processor import GNNProcessorSchema  # noqa: TC001
from .processor import GraphTransformerProcessorSchema  # noqa: TC001
from .processor import NoOpProcessorSchema  # noqa: TC001
from .processor import PointWiseMLPProcessorSchema  # noqa: TC001
from .processor import TransformerProcessorSchema  # noqa: TC001
from .residual import ResidualConnectionSchema

LOGGER = logging.getLogger(__name__)


class DefinedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"
    ANEMOI_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC = "anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiEnsModelEncProcDec"
    ANEMOI_MODEL_HIER_ENC_PROC_DEC = "anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical"
    ANEMOI_MODEL_HIER_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDecHierarchical"
    ANEMOI_TRANSPORT_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.transport_encoder_processor_decoder.AnemoiTransportModelEncProcDec"
    )
    ANEMOI_TRANSPORT_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiTransportModelEncProcDec"
    ANEMOI_TRANSPORT_TEND_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.transport_encoder_processor_decoder.AnemoiTransportTendModelEncProcDec"
    )
    ANEMOI_TRANSPORT_TEND_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiTransportTendModelEncProcDec"


class Model(BaseModel):
    target_: DefinedModels = Field(..., alias="_target_")
    "Model object defined in anemoi.models.model."
    hidden_nodes_name: str | list[str] = Field(examples=["hidden", ["hidden1", "hidden2"]])
    "Name of the hidden nodes. If the model is hierarchical, it can be a list of names for each level."
    latent_skip: bool = Field(default=True)
    "Add skip connection in latent space before/after processor."
    convert_: str = Field("none", alias="_convert_")
    "Keep OmegaConf containers when instantiating — model code uses attribute-style access throughout."


class SparseProjectorSchema(BaseModel):
    num_chunks: PositiveInt = Field(default=1, examples=[1])
    "Number of chunks to use for sparse projection matmuls."


class TransportSourceConfig(BaseModel):
    """Configuration of the starting/source field for transport objectives.

    The defaults map 1:1 to :class:`TransportSourceSettings` in
    ``anemoi.models.transport.settings`` and are restated here so that importing
    the schemas does not pull in the model code.
    """

    kind: Literal["default", "zero", "gaussian", "reference_state"] = "default"
    "Starting field used before the transport objective moves toward the target."
    scale: NonNegativeFloat = Field(default=1.0, examples=[1.0])
    "Multiplier applied to the starting/source field."
    noise_scale: NonNegativeFloat = Field(default=0.0, examples=[0.1])
    "Additional additive Gaussian noise applied to the starting/source field."


class TransportConfig(BaseModel):
    """Configuration of the transport objective, path, conditioning and inference.

    The defaults map 1:1 to :class:`EdmSettings`, :class:`NoiseConditioningSettings`
    and :class:`StochasticInterpolantSettings` in ``anemoi.models.transport.settings``
    and are restated here so that importing the schemas does not pull in the model code.
    """

    objective: Literal["edm_diffusion", "stochastic_interpolant"] = "edm_diffusion"
    "Training and sampling objective used by the transport model."
    sigma_data: PositiveFloat = Field(default=1.0, examples=[1.0])
    "Typical data scale used by EDM diffusion."
    noise_channels: PositiveInt = Field(default=32, examples=[32])
    "Number of channels in the noise or bridge-time embedding."
    noise_cond_dim: PositiveInt = Field(default=16, examples=[16])
    "Size of the conditioning vector passed to conditional layers."
    sigma_max: PositiveFloat = Field(default=100.0, examples=[100.0])
    "Maximum EDM diffusion noise level used during training."
    sigma_min: PositiveFloat = Field(default=0.02, examples=[0.02])
    "Minimum EDM diffusion noise level used during training."
    rho: PositiveFloat = Field(default=7.0, examples=[7.0])
    "Shape parameter for the Karras EDM noise schedule."
    si_alpha_schedule: Literal["linear"] = "linear"
    "Schedule for how strongly the SI bridge keeps the source field."
    si_beta_schedule: Literal["linear", "quadratic"] = "linear"
    "Schedule for how strongly the SI bridge moves toward the target field."
    si_sigma_schedule: Literal["brownian_bridge", "quadratic_bridge"] = "brownian_bridge"
    "Schedule for the SI bridge-noise amplitude."
    source: TransportSourceConfig = Field(default_factory=TransportSourceConfig)
    "Configuration for the starting/source field."
    si_noise_scale: NonNegativeFloat = Field(default=1.0, examples=[1.0])
    "Overall scale of the stochastic-interpolant bridge noise."
    training_condition: dict = Field(default_factory=dict)
    "Distribution used to sample one training noise level or bridge time per sample."
    noise_embedder: dict = Field(default_factory=dict)
    "Hydra configuration for embedding the current noise level or bridge time."
    inference_defaults: dict = Field(default_factory=dict)
    "Default sampler parameters used during inference."


class TransportModel(Model):
    transport: TransportConfig = Field(default_factory=TransportConfig)
    "Transport model objective, path, conditioning, and inference configuration."


class NoOutputMaskSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.NoOutputMask"] = Field(..., alias="_target_")


class Boolean1DSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.Boolean1DMask"] = Field(..., alias="_target_")
    attribute_name: str = Field(example="cutout_mask")


OutputMaskSchemas = Union[NoOutputMaskSchema, Boolean1DSchema]


class EncodersSchema(BaseModel):
    """Encoder schema"""

    source_datasets: list[str] = Field(..., example=["dataset1", "dataset2"])
    "List of datasets for which the encoder is applicable."
    dataset_fusing_strategy: Literal["not_supported"] = Field(default="not_supported")
    "Dataset fusing strategy. Default to 'not_supported'."
    mapper: Union[
        GNNEncoderSchema,
        GraphTransformerEncoderSchema,
        TransformerEncoderSchema,
        PointWiseForwardMapperSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )


class DecodersSchema(BaseModel):
    """Decoder schema"""

    target_datasets: list[str] = Field(..., example=["dataset1", "dataset2"])
    "List of datasets for which the decoder is applicable."
    target_node_features: list[Literal[tuple(sorted(VALID_TARGET_FEATURES))]] = Field(
        default_factory=lambda: ["encoded_data"]
    )
    "Whether to use the encoded latents from the encoder."
    mapper: Union[
        GNNDecoderSchema,
        GraphTransformerDecoderSchema,
        TransformerDecoderSchema,
        PointWiseBackwardMapperSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )


class BaseModelSchema(PydanticBaseModel):
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded"
    sparse_projector: SparseProjectorSchema = Field(default_factory=SparseProjectorSchema)
    "Sparse projection settings."
    model: Model = Field(default_factory=Model)
    "Model schema."
    node_trainable_parameters: dict[str, NonNegativeInt] = Field(examples=[{"data": 8, "hidden": 8}])
    "Learnable node and edge parameters."
    bounding: DatasetDict[list[BoundingSchema]]
    "List of bounding configuration applied in order to the specified variables."
    output_mask: DatasetDict[OutputMaskSchemas]  # !TODO CHECK!
    "Output mask"
    latent_skip: bool = True
    "Add skip connection in latent space before/after processor."
    latent_aggregator: AggregatorSchema
    "Latent aggregator schema."
    processor: Union[
        NoOpProcessorSchema,
        GNNProcessorSchema,
        GraphTransformerProcessorSchema,
        TransformerProcessorSchema,
        PointWiseMLPProcessorSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )
    "Model processor schema."
    encoders: dict[str, EncodersSchema]
    "Model encoders schemas."
    decoders: dict[str, DecodersSchema]
    "Model decoders schemas."
    residual: DatasetDict[ResidualConnectionSchema]
    "Residual connection schema."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to be compiled"
    recompile_limit: PositiveInt = 8
    "How many times torch.compile will recompile a function for a given input shape."

    @model_validator(mode="before")
    @classmethod
    def cast_encoder_decoder_keys_to_str(cls, data: Any) -> Any:
        """Cast encoder/decoder dict keys to str (YAML may parse them as int)."""
        for field in ("encoders", "decoders"):
            if field in data:
                if isinstance(data[field], dict):
                    data[field] = {str(k): v for k, v in data[field].items()}
                elif isinstance(data[field], DictConfig):
                    data[field] = OmegaConf.create({str(k): v for k, v in data[field].items()})
        return data


class NoOpNoiseInjectorSchema(BaseModel):
    """Schema for NoOpNoiseInjector - passes input through unchanged."""

    target_: Literal["anemoi.models.layers.ensemble.NoOpNoiseInjector"] = Field(..., alias="_target_")
    "No-op noise injector class"


class NoiseConditioningSchema(BaseModel):
    """Schema for NoiseConditioning - generates noise for conditioning."""

    target_: Literal["anemoi.models.layers.ensemble.NoiseConditioning"] = Field(..., alias="_target_")
    "Noise conditioning layer class"
    noise_std: NonNegativeInt = Field(example=1)
    "Standard deviation of the noise to be injected."
    noise_channels_dim: NonNegativeInt = Field(example=4)
    "Number of channels in the noise tensor."
    noise_mlp_hidden_dim: NonNegativeInt = Field(example=8)
    "Hidden dimension of the MLP used to process the noise."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"
    noise_matrix: Optional[str] = Field(default=None)
    "Path to the noise projection matrix file (.npz). If None, no projection is applied."
    noise_edges_name: Optional[tuple[str, str, str]] = Field(default=None)
    "Edge type identifier (src, relation, dst) for graph-based noise projection."
    edge_weight_attribute: Optional[str] = Field(default=None)
    "Optional edge attribute name for graph-based noise projection weights."
    row_normalize_noise_matrix: bool = Field(default=False)
    "Whether to row-normalize the noise projection matrix weights."
    autocast: bool = Field(default=False)
    "Whether to use autocast for the noise projection matrix operations."


class NoiseInjectorSchema(BaseModel):
    """Schema for NoiseInjector - injects noise directly into input tensor."""

    target_: Literal["anemoi.models.layers.ensemble.NoiseInjector"] = Field(..., alias="_target_")
    "Noise injector layer class"
    noise_std: NonNegativeInt = Field(example=1)
    "Standard deviation of the noise to be injected."
    noise_channels_dim: NonNegativeInt = Field(example=4)
    "Number of channels in the noise tensor."
    noise_mlp_hidden_dim: NonNegativeInt = Field(example=8)
    "Hidden dimension of the MLP used to process the noise."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


NoiseInjectorUnion = Annotated[
    Union[NoOpNoiseInjectorSchema, NoiseConditioningSchema, NoiseInjectorSchema],
    Field(discriminator="target_"),
]


class SpectrumTableSchema(BaseModel):
    """Tabulated angular power spectrum of one noise channel."""

    degree: list[PositiveFloat] = Field(..., min_length=2, examples=[[1, 2, 4, 8, 16, 32, 64, 128]])
    "Strictly increasing spherical-harmonic degrees, all at least 1."
    sigma2: list[PositiveFloat] = Field(..., min_length=2)
    "Variance of each coefficient at those degrees; interpolated in log-log, only its shape matters."

    @model_validator(mode="after")
    def check_table(self) -> SpectrumTableSchema:
        if len(self.degree) != len(self.sigma2):
            raise ValueError(f"spectrum has {len(self.degree)} degrees but {len(self.sigma2)} values.")
        if any(degree < 1 for degree in self.degree) or any(b <= a for a, b in zip(self.degree, self.degree[1:])):
            raise ValueError("spectrum degrees must be strictly increasing and at least 1.")
        return self


class NoiseChannelSchema(BaseModel):
    """One named input-noise channel: its spectrum and, optionally, the spread that scales it."""

    kT: Optional[PositiveFloat] = Field(default=None, examples=[3.1545e-2])
    "FourCastNet 3 spectrum exp(-kT l(l+1)), kT = (L / 6370 km)^2 / 2 for a length scale L."
    spectrum: Optional[SpectrumTableSchema] = Field(default=None)
    "Tabulated spectrum, e.g. the measured spread spectrum of the variable the channel stands for."
    spread: list[str] = Field(default_factory=list, examples=[["u_850", "v_850"]])
    "Spread variables whose map scales this channel, by base, exact or full name. Empty: never scaled."
    smoothing_km: Optional[PositiveFloat] = Field(default=None)
    "Smoothing length scale of this channel's spread map, overriding the modulation rule."

    @model_validator(mode="after")
    def check_one_spectrum(self) -> NoiseChannelSchema:
        if (self.kT is None) == (self.spectrum is None):
            raise ValueError("a noise channel needs exactly one of 'kT' and 'spectrum'.")
        return self


class SphericalInputNoiseSchema(BaseModel):
    """Schema for SphericalInputNoise - FourCastNet 3 style input perturbation."""

    target_: Literal["anemoi.models.layers.ensemble.SphericalInputNoise"] = Field(..., alias="_target_")
    "Spherical input noise class"
    grid: Union[str, int] = Field(example="n320")
    "Grid the data nodes live on: 'nNNN' reduced Gaussian, 'oNNN' octahedral, or an integer nlat."
    noise: dict = Field(...)
    "Noise field configuration: 'type' (diffusion/white/dummy) plus that type's parameters."
    n_channels: Optional[PositiveInt] = Field(default=None)
    "Number of noise channels appended per input time step: one per noise.kT. Defaults to one, or to len(channels)."
    channels: Optional[dict[str, NoiseChannelSchema]] = Field(default=None)
    "Named channels, each with its own spectrum; replaces noise.kT and needs a diffusion field."
    centered: bool = Field(default=False)
    "Antithetic pairing of ensemble members."
    dataset: Optional[str] = Field(default=None)
    "Dataset whose input the noise is appended to. Required if the model has multiple input datasets."
    default_lambd: float = Field(default=1.0)
    "Default temporal decorrelation rate, dt / 6h in FourCastNet 3."

    @model_validator(mode="after")
    def check_channels(self) -> SphericalInputNoiseSchema:
        if not self.channels:
            return self
        if self.noise.get("type") != "diffusion":
            raise ValueError("named channels set their own spectra, which needs noise.type 'diffusion'.")
        if "kT" in self.noise:
            raise ValueError("give kT per channel under 'channels', not under 'noise'.")
        if self.n_channels is not None and self.n_channels != len(self.channels):
            raise ValueError(f"n_channels={self.n_channels} but {len(self.channels)} channels are defined.")
        if self.target_ == "anemoi.models.layers.ensemble.SphericalInputNoise":
            spread = sorted(name for name, channel in self.channels.items() if channel.spread)
            if spread:
                raise ValueError(f"channels {spread} name 'spread', which needs SphericalInputConditionedNoise.")
        return self


class BandFilterSchema(BaseModel):
    """Per-channel low-pass applied to each modulated noise channel."""

    quantile: float = Field(default=0.99, gt=0.0, le=1.0)
    "Share of the channel's own variance below its band edge."
    taper: NonNegativeFloat = Field(default=0.25)
    "Raised-cosine roll-off beyond the band edge, as a fraction of the edge degree. 0 gives a hard cut."
    preserve_variance: bool = Field(default=True)
    "Rescale each filtered channel to its variance before filtering: the filter moves energy, never removes it."


class StdModulationSchema(BaseModel):
    """How ensemble spread fields scale the input noise channels that name them."""

    enabled: bool = Field(default=True)
    "Scale the noise. When False the spread inputs are still dropped from the encoder, giving a matched baseline."
    variable_prefix: str = Field(default="std_")
    "Prefix of the spread variables in the input dataset."
    source: Literal["eda_stdev"] = Field(default="eda_stdev")
    "Kind of spread field the variables hold."
    reference: Literal["climatology", "sample_mean"] = Field(default="climatology")
    "Divide each spread by its dataset mean (keeps the level of each era), or by each sample's own global mean."
    normalizer: Optional[Literal["none", "std", "max"]] = Field(default=None)
    "How the data config normalises the spread variables; needed by reference 'climatology'."
    rescale: Optional[Literal["climatology", "sample_rms", "none"]] = Field(default=None)
    "Fixed constant per channel (climatology), unit mean square per sample (sample_rms) or none. Follows reference."
    smoothing_km: Optional[PositiveFloat] = Field(default=100.0)
    "Default smoothing length scale of the multipliers. None disables smoothing."
    smoothing_km_by_variable: dict[str, PositiveFloat] = Field(default_factory=dict, examples=[{"t": 200, "z": 400}])
    "Smoothing length scale per base or exact variable name; a channel takes the coarsest of its variables."
    smooth_to_channel: bool = Field(default=False)
    "Smooth each kT channel's multiplier at the channel's own scale instead."
    clip: Optional[tuple[NonNegativeFloat, PositiveFloat]] = Field(default=(0.05, 10.0))
    "Bounds on the multiplier, in multiples of the reference. None disables clipping."
    area_weight: bool = Field(default=True)
    "Weight spatial means by grid-cell area (Gauss-Legendre quadrature) rather than per point."
    band_filter: BandFilterSchema = Field(default_factory=BandFilterSchema)
    "Low-pass returning each scaled channel to its own scale band."

    @model_validator(mode="after")
    def check_modes(self) -> StdModulationSchema:
        if self.rescale is None:
            self.rescale = "climatology" if self.reference == "climatology" else "sample_rms"
        if self.rescale == "climatology" and self.reference != "climatology":
            raise ValueError("rescale 'climatology' needs reference 'climatology'.")
        if self.enabled and self.reference == "climatology" and self.normalizer is None:
            raise ValueError(
                "reference 'climatology' needs normalizer: how the data config normalises the spread variables."
            )
        return self


class SphericalInputConditionedNoiseSchema(SphericalInputNoiseSchema):
    """Schema for SphericalInputConditionedNoise - FourCastNet 3 noise scaled channel by channel by analysis spread."""

    target_: Literal["anemoi.models.layers.ensemble.SphericalInputConditionedNoise"] = Field(..., alias="_target_")
    "Spread-conditioned spherical input noise class"
    channels: dict[str, NoiseChannelSchema] = Field(..., min_length=1)
    "Named channels, each with its own spectrum and optionally the spread variables that scale it."
    modulation: StdModulationSchema = Field(default_factory=StdModulationSchema)
    "How the spread scales the channels."

    @model_validator(mode="after")
    def check_spread_channels(self) -> SphericalInputConditionedNoiseSchema:
        if not self.modulation.enabled:
            return self
        spread = {name: channel for name, channel in self.channels.items() if channel.spread}
        if not spread:
            raise ValueError("no channel names 'spread' variables: give some, or set modulation.enabled: False.")
        if self.modulation.smooth_to_channel:
            tabulated = sorted(name for name, ch in spread.items() if ch.kT is None and ch.smoothing_km is None)
            if tabulated:
                raise ValueError(f"smooth_to_channel needs kT channels; give {tabulated} a smoothing_km.")
        return self


SphericalInputNoiseUnion = Annotated[
    Union[SphericalInputNoiseSchema, SphericalInputConditionedNoiseSchema],
    Field(discriminator="target_"),
]


class EnsModelSchema(BaseModelSchema):
    noise_injector: NoiseInjectorUnion = Field(...)
    "Noise injection configuration. Use NoOpNoiseInjector to disable, NoiseConditioning for conditioning, or NoiseInjector for direct injection."
    input_noise: Optional[SphericalInputNoiseUnion] = Field(default=None)
    "FourCastNet 3 style spherical input perturbation, concatenated to the encoder input. None disables it."
    condition_on_residual: bool = Field(default=False)
    "Whether to condition the noise injection on the residual connection."


class TransportModelSchema(BaseModelSchema):
    model: TransportModel = Field(default_factory=TransportModel)
    "Transport model schema."

    @model_validator(mode="after")
    def validate_no_bounding_for_transport(self) -> "TransportModelSchema":
        if self.bounding:
            if "datasets" in self.bounding:
                for dataset_name, bounding_list in self.bounding["datasets"].items():
                    if (bounding_list is not None) and len(bounding_list) > 0:
                        msg = (
                            "Transport models do not support bounding layers. "
                            f"Found {len(bounding_list)} bounding configuration(s) for dataset '{dataset_name}'. "
                            f"Please remove all bounding configurations for transport models."
                        )
                        raise ValueError(msg)
            elif len(self.bounding) > 0:
                msg = (
                    "Transport models do not support bounding layers. "
                    f"Found {len(self.bounding)} bounding configuration(s). "
                    f"Please remove all bounding configurations for transport models."
                )
                raise ValueError(msg)
        return self


class TransportTendModelSchema(TransportModelSchema):
    condition_on_residual: bool = Field(default=False)
    "Whether to condition the noise injection on the residual connection."


class HierarchicalModelSchema(BaseModelSchema):
    enable_hierarchical_level_processing: bool = Field(default=False)
    "Toggle to do message passing at every downscaling and upscaling step"
    level_process_num_layers: NonNegativeInt = Field(default=1)
    "Number of message passing steps at each level"
    upscale_mapper: Union[
        GNNEncoderSchema,
        GraphTransformerEncoderSchema,
        TransformerEncoderSchema,
        PointWiseForwardMapperSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )
    "Mapper used to upscale from a lower level to a higher level in the hierarchy."
    downscale_mapper: Union[
        GNNDecoderSchema,
        GraphTransformerDecoderSchema,
        TransformerDecoderSchema,
        PointWiseBackwardMapperSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )
    "Mapper used to downscale from a higher level to a lower level in the hierarchy."

    @model_validator(mode="before")
    @classmethod
    def default_num_channels_in_hierarchical_mapper(cls, data: Any) -> Any:
        """Allow num_channels to be omitted.

        It will be set at model build time.
        """
        for mapper_field in ("upscale_mapper", "downscale_mapper"):
            if mapper_field in data:
                mapper = data[mapper_field]
                if isinstance(data, dict):
                    mapper["num_channels"] = 1
                elif isinstance(data, DictConfig):
                    OmegaConf.update(mapper, "num_channels", 1, force_add=True)
        return data


ModelSchema = Union[
    BaseModelSchema,
    EnsModelSchema,
    HierarchicalModelSchema,
    TransportModelSchema,
    TransportTendModelSchema,
]

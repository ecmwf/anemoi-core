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

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from anemoi.models.transport.settings import EdmSettings
from anemoi.models.transport.settings import NoiseConditioningSettings
from anemoi.models.transport.settings import StochasticInterpolantSettings
from anemoi.models.transport.settings import TransportSourceSettings
from anemoi.utils.schemas import BaseModel

from .aggregator import AggregatorSchema  # noqa: TC001
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
    ANEMOI_MODEL_AUTOENCODER = "anemoi.models.models.autoencoder.AnemoiModelAutoEncoder"
    ANEMOI_MODEL_AUTOENCODER_SHORT = "anemoi.models.models.AnemoiModelAutoEncoder"
    ANEMOI_MODEL_HIER_AUTOENCODER = "anemoi.models.models.autoencoder.AnemoiModelHierarchicalAutoEncoder"
    ANEMOI_MODEL_HIER_AUTOENCODER_SHORT = "anemoi.models.models.AnemoiModelHierarchicalAutoEncoder"


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
    kind: Literal["default", "zero", "gaussian", "reference_state"] = TransportSourceSettings.kind
    "Starting field used before the transport objective moves toward the target."
    scale: NonNegativeFloat = Field(default=TransportSourceSettings.scale, examples=[TransportSourceSettings.scale])
    "Multiplier applied to the starting/source field."
    noise_scale: NonNegativeFloat = Field(default=TransportSourceSettings.noise_scale, examples=[0.1])
    "Additional additive Gaussian noise applied to the starting/source field."


class TransportConfig(BaseModel):
    objective: Literal["edm_diffusion", "stochastic_interpolant"] = "edm_diffusion"
    "Training and sampling objective used by the transport model."
    sigma_data: PositiveFloat = Field(default=EdmSettings.sigma_data, examples=[EdmSettings.sigma_data])
    "Typical data scale used by EDM diffusion."
    noise_channels: PositiveInt = Field(
        default=NoiseConditioningSettings.channels,
        examples=[NoiseConditioningSettings.channels],
    )
    "Number of channels in the noise or bridge-time embedding."
    noise_cond_dim: PositiveInt = Field(
        default=NoiseConditioningSettings.cond_dim,
        examples=[NoiseConditioningSettings.cond_dim],
    )
    "Size of the conditioning vector passed to conditional layers."
    sigma_max: PositiveFloat = Field(default=EdmSettings.sigma_max, examples=[EdmSettings.sigma_max])
    "Maximum EDM diffusion noise level used during training."
    sigma_min: PositiveFloat = Field(default=EdmSettings.sigma_min, examples=[EdmSettings.sigma_min])
    "Minimum EDM diffusion noise level used during training."
    rho: PositiveFloat = Field(default=EdmSettings.rho, examples=[EdmSettings.rho])
    "Shape parameter for the Karras EDM noise schedule."
    si_alpha_schedule: Literal["linear"] = StochasticInterpolantSettings.alpha_schedule
    "Schedule for how strongly the SI bridge keeps the source field."
    si_beta_schedule: Literal["linear", "quadratic"] = StochasticInterpolantSettings.beta_schedule
    "Schedule for how strongly the SI bridge moves toward the target field."
    si_sigma_schedule: Literal["brownian_bridge", "quadratic_bridge"] = StochasticInterpolantSettings.sigma_schedule
    "Schedule for the SI bridge-noise amplitude."
    source: TransportSourceConfig = Field(default_factory=TransportSourceConfig)
    "Configuration for the starting/source field."
    si_noise_scale: NonNegativeFloat = Field(
        default=StochasticInterpolantSettings.noise_scale,
        examples=[StochasticInterpolantSettings.noise_scale],
    )
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


class TrainableParameters(PydanticBaseModel):
    data: NonNegativeInt = Field(example=8)
    "Size of the learnable data node tensor. Default to 8."
    hidden: NonNegativeInt = Field(example=8)
    "Size of the learnable hidden node tensor. Default to 8."


class ReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.ReluBounding"] = Field(..., alias="_target_")
    "Relu bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the Relu method."


class LeakyReluBoundingSchema(ReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky Relu bounding object defined in anemoi.models.layers.bounding."


class FractionBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.FractionBounding"] = Field(..., alias="_target_")
    "Fraction bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh fraction method."
    min_val: float
    "The minimum value for the HardTanh activation. Correspond to the minimum fraction of the total_var."
    max_val: float
    "The maximum value for the HardTanh activation. Correspond to the maximum fraction of the total_var."
    total_var: str
    "Variable from which the secondary variables are derived. \
    For example, convective precipitation should be a fraction of total precipitation."


class LeakyFractionBoundingSchema(FractionBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyFractionBounding"] = Field(..., alias="_target_")
    "Leaky fraction bounding object defined in anemoi.models.layers.bounding."


class HardtanhBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.HardtanhBounding"] = Field(..., alias="_target_")
    "Hard tanh bounding method function from anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh method."
    min_val: float
    "The minimum value for the HardTanh activation."
    max_val: float
    "The maximum value for the HardTanh activation."


class LeakyHardtanhBoundingSchema(HardtanhBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyHardtanhBounding"] = Field(..., alias="_target_")
    "Leaky hard tanh bounding method function from anemoi.models.layers.bounding."


class NormalizedReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.NormalizedReluBounding"] = Field(..., alias="_target_")
    variables: list[str]
    min_val: list[float]
    normalizer: list[str]

    @model_validator(mode="after")
    def check_num_normalizers_and_min_val_matches_num_variables(
        self,
    ) -> NormalizedReluBoundingSchema:
        error_msg = f"""{self.__class__} requires that number of normalizers ({len(self.normalizer)}) or
        match the number of variables ({len(self.variables)})"""
        assert len(self.normalizer) == len(self.variables), error_msg
        error_msg = f"""{self.__class__} requires that number of min_val ({len(self.min_val)}) or  match
        the number of variables ({len(self.variables)})"""
        assert len(self.min_val) == len(self.variables), error_msg
        return self


class NormalizedLeakyReluBoundingSchema(NormalizedReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.NormalizedLeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky normalized Relu bounding object defined in anemoi.models.layers.bounding."


Bounding = Annotated[
    Union[
        ReluBoundingSchema,
        LeakyReluBoundingSchema,
        FractionBoundingSchema,
        LeakyFractionBoundingSchema,
        HardtanhBoundingSchema,
        LeakyHardtanhBoundingSchema,
        NormalizedReluBoundingSchema,
        NormalizedLeakyReluBoundingSchema,
    ],
    Field(discriminator="target_"),
]


class NoOutputMaskSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.NoOutputMask"] = Field(..., alias="_target_")


class Boolean1DSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.Boolean1DMask"] = Field(..., alias="_target_")
    attribute_name: str = Field(example="cutout_mask")


OutputMaskSchemas = Union[NoOutputMaskSchema, Boolean1DSchema]


class EncodersSchema(BaseModel):
    """Encoder schema"""

    datasets: list[str] = Field(..., example=["dataset1", "dataset2"])
    "List of datasets for which the encoder is applicable."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(default_factory=list)
    "Edge attributes to consider in the model component features."
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

    datasets: list[str] = Field(..., example=["dataset1", "dataset2"])
    "List of datasets for which the decoder is applicable."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(default_factory=list)
    "List of edge attributes for the sub-graph."
    input_target_features: list[
        Literal["coordinates", "forcings", "prognostics", "trainable_parameters", "encoded_data"]
    ] = Field(default_factory=lambda: ["encoded_data"])
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
    num_channels: NonNegativeInt = Field(example=512)
    "Feature tensor size in the hidden space."
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded"
    sparse_projector: SparseProjectorSchema = Field(default_factory=SparseProjectorSchema)
    "Sparse projection settings."
    model: Model = Field(default_factory=Model)
    "Model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters."
    bounding: list[Bounding]
    "List of bounding configuration applied in order to the specified variables."
    output_mask: OutputMaskSchemas  # !TODO CHECK!
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
    residual: ResidualConnectionSchema = Field(
        ...,
        discriminator="target_",
    )
    "Residual connection schema."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to be compiled"
    recompile_limit: PositiveInt = 8
    "How many times torch.compile will recompile a function for a given input shape."


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


class EnsModelSchema(BaseModelSchema):
    noise_injector: NoiseInjectorUnion = Field(...)
    "Noise injection configuration. Use NoOpNoiseInjector to disable, NoiseConditioning for conditioning, or NoiseInjector for direct injection."
    condition_on_residual: bool = Field(default=False)
    "Whether to condition the noise injection on the residual connection."


class TransportModelSchema(BaseModelSchema):
    model: TransportModel = Field(default_factory=TransportModel)
    "Transport model schema."

    @model_validator(mode="after")
    def validate_no_bounding_for_transport(self) -> "TransportModelSchema":
        if self.bounding:
            msg = (
                "Transport models do not support bounding layers. "
                f"Found {len(self.bounding)} bounding configuration(s). "
                "Please remove all bounding configurations for transport models."
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


ModelSchema = Union[
    BaseModelSchema,
    EnsModelSchema,
    HierarchicalModelSchema,
    TransportModelSchema,
    TransportTendModelSchema,
]

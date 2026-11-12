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
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from anemoi.models.models.target_features import VALID_TARGET_FEATURES
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
    hidden_nodes_name: str | list[str] | dict[str, str] = Field(
        examples=["hidden", ["hidden1", "hidden2"], {"dataset1": "hidden1", "dataset2": "hidden2"}]
    )
    "Name of the hidden nodes, per-dataset names, or names for each level of a hierarchical model."
    latent_skip: bool = Field(default=True)
    "Add skip connection in latent space before/after processor."
    convert_: str = Field("none", alias="_convert_")
    "Keep OmegaConf containers when instantiating — model code uses attribute-style access throughout."


class SparseProjectorSchema(BaseModel):
    num_chunks: PositiveInt = Field(default=1, examples=[1])
    "Number of chunks to use for sparse projection matmuls."


class TransportConfig(BaseModel):
    """Configuration of the transport objective, path, conditioning and inference.

    The defaults map 1:1 to :class:`EdmSettings` and :class:`NoiseConditioningSettings`
    in ``anemoi.models.transport.settings`` and are restated here so that importing the
    schemas does not pull in the model code.
    """

    objective: Literal["edm_diffusion", "flow_matching"] = "edm_diffusion"
    "Training and sampling objective used by the transport model."
    sigma_data: PositiveFloat = Field(default=1.0, examples=[1.0])
    "Typical data scale used by EDM diffusion."
    noise_channels: PositiveInt = Field(default=32, examples=[32])
    "Number of channels in the noise-level or time embedding."
    noise_cond_dim: PositiveInt = Field(default=16, examples=[16])
    "Size of the conditioning vector passed to conditional layers."
    sigma_max: PositiveFloat = Field(default=100.0, examples=[100.0])
    "Maximum EDM diffusion noise level used during training."
    sigma_min: PositiveFloat = Field(default=0.02, examples=[0.02])
    "Minimum EDM diffusion noise level used during training."
    rho: PositiveFloat = Field(default=7.0, examples=[7.0])
    "Shape parameter for the Karras EDM noise schedule."
    training_condition: dict = Field(default_factory=dict)
    "Distribution used to sample one training noise level or time per sample."
    noise_embedder: dict = Field(default_factory=dict)
    "Hydra configuration for embedding the current noise level or time."
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


class NoOpNodeEmbedderSchema(BaseModel):
    """Pass-through node embedder: the encoder mapper takes each source dataset's input as is."""

    target_: Literal["anemoi.models.layers.node_embedder.NoOpNodeEmbedder"] = Field(..., alias="_target_")


class LinearNodeEmbedderSchema(BaseModel):
    """Linear node embedder: one linear projection per source dataset onto ``num_channels``."""

    target_: Literal["anemoi.models.layers.node_embedder.LinearNodeEmbedder"] = Field(..., alias="_target_")
    num_channels: PositiveInt = Field(example=512)
    "Number of channels the source datasets are projected to; the input channels of the encoder mapper."
    layer_kernels: Union[dict[str, dict], None] = Field(default=None)
    "Layer implementations for the projections. Defaults to the encoder mapper's layer_kernels."


NodeEmbedderSchemas = Annotated[Union[NoOpNodeEmbedderSchema, LinearNodeEmbedderSchema], Field(discriminator="target_")]


class EncodersSchema(BaseModel):
    """Encoder schema"""

    source_datasets: list[str] = Field(..., example=["dataset1", "dataset2"])
    "List of datasets for which the encoder is applicable."
    dataset_fusing_strategy: Literal["none", "sequential", "joint"] = Field(default="none")
    """How the source datasets of this encoder are combined.

    - 'none' (default): no fusion; the node embedder is a ``NoOpNodeEmbedder``, so the mapper takes the
      assembled input of one source dataset directly. Normally ``source_datasets`` lists one dataset; several are
      allowed only if they have the same number of input channels and at most one of them is present in
      any batch (e.g. multi-domain training with one domain per batch).
    - 'sequential': each source dataset is projected by a projecting ``node_embedder`` (e.g.
      ``LinearNodeEmbedder``) onto
      ``node_embedder.num_channels`` and encoded in its own pass with the shared mapper weights, in
      the order listed in ``source_datasets``. The latents are combined by the latent
      aggregator. Source datasets may differ in their number of input channels.
    - 'joint': planned, not yet supported.
    """
    node_embedder: Optional[NodeEmbedderSchemas] = Field(
        default_factory=lambda: NoOpNodeEmbedderSchema(_target_="anemoi.models.layers.node_embedder.NoOpNodeEmbedder")
    )
    """Per-dataset embedding onto the mapper input channels.

    'none' uses ``NoOpNodeEmbedder`` (the default; ``null`` is read as it too, and any other node embedder is
    ignored). 'sequential' needs a projecting one such as ``LinearNodeEmbedder``. A node embedder embeds node
    features only: the encoder edges of all source datasets must still carry the same number of edge features."""
    mapper: Union[
        GNNEncoderSchema,
        GraphTransformerEncoderSchema,
        TransformerEncoderSchema,
        PointWiseForwardMapperSchema,
    ] = Field(
        ...,
        discriminator="target_",
    )

    @model_validator(mode="after")
    def check_fusing_strategy(self) -> EncodersSchema:
        """Reject the planned 'joint' strategy and a 'sequential' encoder without a projecting node embedder."""
        if self.dataset_fusing_strategy == "joint":
            msg = "dataset_fusing_strategy 'joint' is planned but not yet supported; use 'none' or 'sequential'."
            raise ValueError(msg)
        pass_through = self.node_embedder is None or isinstance(self.node_embedder, NoOpNodeEmbedderSchema)
        if self.dataset_fusing_strategy == "sequential" and pass_through:
            msg = (
                "dataset_fusing_strategy 'sequential' needs a projecting node embedder (not NoOpNodeEmbedder), e.g. "
                "node_embedder: {_target_: anemoi.models.layers.node_embedder.LinearNodeEmbedder, num_channels: 512}."
            )
            raise ValueError(msg)
        if self.dataset_fusing_strategy == "none" and not pass_through:
            LOGGER.warning(
                "Encoder with source_datasets %s has dataset_fusing_strategy 'none'; its node embedder %s is ignored "
                "and NoOpNodeEmbedder is used instead.",
                self.source_datasets,
                self.node_embedder.target_,
            )
        return self


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
    output_mask: DatasetDict[OutputMaskSchemas]  # !TODO CHECK!
    "Output mask"
    latent_skip: bool = True
    "Add skip connection in latent space before/after processor."
    latent_aggregator: AggregatorSchema
    "Latent aggregator schema."
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

    # Checked here rather than in HierarchicalModelSchema: a hierarchical config also validates as
    # BaseModelSchema, so a check that only HierarchicalModelSchema runs would be skipped.
    @model_validator(mode="after")
    def validate_hidden_nodes_name(self) -> "BaseModelSchema":
        hierarchical_models = {
            DefinedModels.ANEMOI_MODEL_HIER_ENC_PROC_DEC,
            DefinedModels.ANEMOI_MODEL_HIER_ENC_PROC_DEC_SHORT,
        }
        hidden_nodes_name = self.model.hidden_nodes_name
        if self.model.target_ in hierarchical_models:
            if not isinstance(hidden_nodes_name, list) or not hidden_nodes_name:
                msg = "Hierarchical models need a non-empty list of hidden node names, one per level."
                raise ValueError(msg)
        elif isinstance(hidden_nodes_name, list):
            msg = (
                "Only hierarchical models take a list of hidden node names. "
                "Use a single name, or a mapping from dataset to hidden node name."
            )
            raise ValueError(msg)
        return self


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

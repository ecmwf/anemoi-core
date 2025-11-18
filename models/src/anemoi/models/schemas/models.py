# (C) Copyright 2024 ECMWF.
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
from typing import Literal
from typing import Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel

from .decoder import GNNDecoderSchema  # noqa: TC001
from .decoder import GraphTransformerDecoderSchema  # noqa: TC001
from .decoder import TransformerDecoderSchema  # noqa: TC001
from .encoder import GNNEncoderSchema  # noqa: TC001
from .encoder import GraphTransformerEncoderSchema  # noqa: TC001
from .encoder import TransformerEncoderSchema  # noqa: TC001
from .processor import GNNProcessorSchema  # noqa: TC001
from .processor import GraphTransformerProcessorSchema  # noqa: TC001
from .processor import PointWiseMLPProcessorSchema  # noqa: TC001
from .processor import TransformerProcessorSchema  # noqa: TC001

LOGGER = logging.getLogger(__name__)


class DefinedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"
    ANEMOI_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC = "anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiEnsModelEncProcDec"
    ANEMOI_MODEL_ENC_HIERPROC_DEC = "anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical"
    ANEMOI_MODEL_ENC_HIERPROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDecHierarchical"
    ANEMOI_MODEL_INTERPENC_PROC_DEC = "anemoi.models.models.interpolator.AnemoiModelEncProcDecInterpolator"
    ANEMOI_MODEL_INTERPENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDecInterpolator"
    ANEMOI_DIFFUSION_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.diffusion_encoder_processor_decoder.AnemoiDiffusionModelEncProcDec"
    )
    ANEMOI_DIFFUSION_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiDiffusionModelEncProcDec"
    ANEMOI_DIFFUSION_TEND_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.diffusion_encoder_processor_decoder.AnemoiDiffusionTendModelEncProcDec"
    )
    ANEMOI_DIFFUSION_TEND_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiDiffusionTendModelEncProcDec"


class Model(BaseModel):
    target_: DefinedModels = Field(..., alias="_target_")
    "Model object defined in anemoi.models.model."
    convert_: str = Field("all", alias="_convert_")
    "The target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."


class DiffusionModel(Model):
    diffusion: DiffusionSchema = Field(default=None)
    "Diffusion configuration for diffusion models"


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
    def check_num_normalizers_and_min_val_matches_num_variables(self) -> NormalizedReluBoundingSchema:
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
    nodes_name: str = Field(examples="data")
    attribute_name: str = Field(example="cutout_mask")


OutputMaskSchemas = Union[NoOutputMaskSchema, Boolean1DSchema]


class DiffusionSchema(BaseModel):
    sigma_data: PositiveFloat = Field(default=1.0, examples=[1.0])
    "Data scaling parameter"
    noise_channels: PositiveInt = Field(default=32, examples=[32])
    "Number of channels for noise embedding"
    noise_cond_dim: PositiveInt = Field(default=16, examples=[16])
    "Dimension of noise conditioning"
    sigma_max: PositiveFloat = Field(default=100.0, examples=[100.0])
    "Maximum noise level for training"
    sigma_min: PositiveFloat = Field(default=0.02, examples=[0.02])
    "Minimum noise level for training"
    rho: PositiveFloat = Field(default=7.0, examples=[7.0])
    "Karras schedule parameter for training noise distribution"
    noise_embedder: dict = Field(default_factory=dict)
    "Noise embedder configuration with _target_ for Hydra instantiation"
    inference_defaults: dict = Field(default_factory=dict)
    "Default parameters for inference sampling"


class BaseModelSchema(PydanticBaseModel):
    num_channels: NonNegativeInt = Field(example=512)
    "Feature tensor size in the hidden space."
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded"
    model: Model = Field(default_factory=Model)
    "Model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters."
    bounding: list[Bounding]
    "List of bounding configuration applied in order to the specified variables."
    output_mask: OutputMaskSchemas  # !TODO CHECK!
    "Output mask"
    latent_skip: bool = True
    "Add skip connection in latent space before/after processor. Currently only in interpolator."
    grid_skip: Union[int, None] = 0  # !TODO set default to -1 if added to standard forecaster.
    "Index of grid residual connection, or use none. Currently only in interpolator."
    processor: Union[
        GNNProcessorSchema, GraphTransformerProcessorSchema, TransformerProcessorSchema, PointWiseMLPProcessorSchema
    ] = Field(
        ...,
        discriminator="target_",
    )
    "GNN processor schema."
    encoder: Union[GNNEncoderSchema, GraphTransformerEncoderSchema, TransformerEncoderSchema] = Field(
        ...,
        discriminator="target_",
    )
    "GNN encoder schema."
    decoder: Union[GNNDecoderSchema, GraphTransformerDecoderSchema, TransformerDecoderSchema] = Field(
        ...,
        discriminator="target_",
    )
    "GNN decoder schema."


class NoiseInjectorSchema(BaseModel):
    target_: Literal["anemoi.models.layers.ensemble.NoiseConditioning"] = Field(..., alias="_target_")
    "Noise injection layer class"
    noise_std: NonNegativeInt = Field(example=1)
    "Standard deviation of the noise to be injected."
    noise_channels_dim: NonNegativeInt = Field(example=4)
    "Number of channels in the noise tensor."
    noise_mlp_hidden_dim: NonNegativeInt = Field(example=8)
    "Hidden dimension of the MLP used to process the noise."
    inject_noise: bool = Field(default=True)
    "Whether to inject noise or not."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


class EnsModelSchema(BaseModelSchema):
    noise_injector: NoiseInjectorSchema = Field(default_factory=list)
    "Settings related to custom kernels for encoder processor and decoder blocks"


class DiffusionModelSchema(BaseModelSchema):
    model: DiffusionModel = Field(default_factory=DiffusionModel)
    "Diffusion Model schema"

    @model_validator(mode="after")
    def validate_no_bounding_for_diffusion(self) -> "DiffusionModelSchema":
        if self.bounding:
            msg = (
                "Diffusion models do not support bounding layers. "
                f"Found {len(self.bounding)} bounding configuration(s). "
                "Please remove all bounding configurations for diffusion models."
            )
            raise ValueError(msg)
        return self


class HierarchicalModelSchema(BaseModelSchema):
    enable_hierarchical_level_processing: bool = Field(default=False)
    "Toggle to do message passing at every downscaling and upscaling step"
    level_process_num_layers: NonNegativeInt = Field(default=1)
    "Number of message passing steps at each level"


ModelSchema = Union[BaseModelSchema, EnsModelSchema, HierarchicalModelSchema, DiffusionModelSchema]

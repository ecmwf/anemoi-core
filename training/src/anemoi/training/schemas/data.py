# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel


class NormalizerSchema(BaseModel):
    default: Union[str, None] = Field(literals=["mean-std", "std", "min-max", "max", "none"])
    """Normalizer default method to apply"""
    remap: Union[dict[str, str], None] = Field(default_factory=dict)
    """Dictionary for remapping variables"""
    std: Union[list[str], None] = Field(default_factory=list)
    """Variables to normalise with std"""
    mean_std: Union[list[str], None] = Field(default_factory=list, alias="mean-std")
    """Variables to mormalize with mean-std"""
    min_max: Union[list[str], None] = Field(default_factory=list, alias="min-max")
    """Variables to normalize with min-max."""
    max: Union[list[str], None] = Field(default_factory=list)
    """Variables to normalize with max."""
    none: Union[list[str], None] = Field(default_factory=list)
    """Variables not to be normalized."""


class ImputerSchema(BaseModel):
    default: str = Field(literals=["none", "mean", "stdev"])
    "Imputer default method to apply."
    maximum: Union[list[str], None]
    minimum: Union[list[str], None]
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be imputed."


class ConstantImputerSchema(RootModel[dict[Any, Any]]):
    """Schema for constant imputer.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    1:
        - y
    5.0:
        - x
    3.14:
        - q
    none:
        - z
        - other
    ```
    """

    @field_validator("root")
    @classmethod
    def validate_entries(cls, values: dict[Union[int, float, str], Union[str, list[str]]]) -> dict[Any, Any]:

        for k, v in values.items():
            if k == "default":
                if not isinstance(v, (int, float)):
                    if v is None or v == "none" or v == "None":
                        continue
                    msg = f'"default" must map to a float or None, got {type(v).__name__}'
                    raise TypeError(msg)
            elif k == "none":
                if not isinstance(v, list) or not all(isinstance(i, str) for i in v):
                    msg = f'"none" must map to a list of strings, got {v}'
                    raise TypeError(msg)

            elif isinstance(k, (int, float)):
                if not isinstance(v, list) or not all(isinstance(i, str) for i in v):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)

            # Reject all other keys
            else:
                msg = f'Key "{k}" must be either a number, "none" or "default", got {type(k).__name__}'
                raise TypeError(msg)

        return values


class PostprocessorSchema(BaseModel):
    default: str = Field(literals=["none", "relu", "hardtanh"])
    "Postprocessor default method to apply."
    relu: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with relu."
    hardtanh: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with hardtanh."
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be postprocessed."


class NormalizedReluPostprocessorSchema(BaseModel):
    default: Union[str, float] = Field(default=None)
    "Postprocessor default method to apply."
    # TODO(how): float possible
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be postprocessed."


class ConditionalZeroPostprocessorSchema(BaseModel):
    default: Union[str, float] = Field(default=None)
    "Postprocessor default method to apply."
    remap: str
    "Variable to be used as conditional."
    # TODO(how): float possible
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be postprocessed."


class RemapperSchema(BaseModel):
    default: str = Field(literals=["none", "log1p", "sqrt", "boxcox"])
    "Remapper default method to apply."
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be remapped."


class PreprocessorTarget(str, Enum):
    normalizer = "anemoi.models.preprocessing.normalizer.InputNormalizer"
    imputer = "anemoi.models.preprocessing.imputer.InputImputer"
    const_imputer = "anemoi.models.preprocessing.imputer.ConstantImputer"
    remapper = "anemoi.models.preprocessing.remapper.Remapper"
    postprocessor = "anemoi.models.preprocessing.postprocessor.Postprocessor"
    conditional_zero_postprocessor = "anemoi.models.preprocessing.postprocessor.ConditionalZeroPostprocessor"
    normalized_relu_postprocessor = "anemoi.models.preprocessing.postprocessor.NormalizedReluPostprocessor"


target_to_schema = {
    PreprocessorTarget.normalizer: NormalizerSchema,
    PreprocessorTarget.imputer: ImputerSchema,
    PreprocessorTarget.const_imputer: ConstantImputerSchema,
    PreprocessorTarget.remapper: RemapperSchema,
    PreprocessorTarget.postprocessor: PostprocessorSchema,
    PreprocessorTarget.conditional_zero_postprocessor: ConditionalZeroPostprocessorSchema,
    PreprocessorTarget.normalized_relu_postprocessor: NormalizedReluPostprocessorSchema,
}


class PreprocessorSchema(BaseModel):
    target_: PreprocessorTarget = Field(..., alias="_target_")
    "Processor object from anemoi.models.preprocessing.[normalizer|imputer|remapper]."
    config: Union[
        NormalizerSchema,
        ImputerSchema,
        ConstantImputerSchema,
        RemapperSchema,
        PostprocessorSchema,
        ConditionalZeroPostprocessorSchema,
        NormalizedReluPostprocessorSchema,
    ]
    "Target schema containing processor methods."
    normalizer: Union[str, None] = Field(default=None, literals=["none", "mean-std", "std", "min-max", "max"])
    "Normalizer method to apply before normalized relu."

    @model_validator(mode="after")
    def schema_consistent_with_target(self) -> PreprocessorSchema:
        if self.target_ not in target_to_schema or target_to_schema[self.target_] != self.config.__class__:
            error_msg = f"Schema {self.config.__class__} does not match target {self.target_}"
            raise ValidationError(error_msg)
        return self


class DataSchema(PydanticBaseModel):
    """A class used to represent the overall configuration of the dataset.

    Attributes
    ----------
    format : str
        The format of the data.
    resolution : str
        The resolution of the data.
    frequency : str
        The frequency of the data.
    timestep : str
        The timestep of the data.
    forcing : List[str]
        The list of features used as forcing to generate the forecast state.
    diagnostic : List[str]
        The list of features that are only part of the forecast state.
    processors : Dict[str, Processor]
        The Processors configuration.
    num_features : Optional[int]
        The number of features in the forecast state. To be set in the code.
    """

    format: str = Field(example=None)
    "Format of the data."
    frequency: str = Field(example=None)
    "Time frequency requested from the dataset."
    timestep: str = Field(example=None)
    "Time step of model (must be multiple of frequency)."
    processors: dict[str, PreprocessorSchema]
    "Layers of model performing computation on latent space. \
            Processors including imputers and normalizers are applied in order of definition."
    forcing: list[str]
    "Features that are not part of the forecast state but are used as forcing to generate the forecast state."
    diagnostic: list[str]
    "Features that are only part of the forecast state and are not used as an input to the model."
    num_features: Union[int, None]
    "Number of features in the forecast state. To be set in the code."

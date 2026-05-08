# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterable
from enum import Enum
from typing import Any
from typing import Union

from pydantic import Field
from pydantic import RootModel
from pydantic import TypeAdapter
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
    maximum: Union[list[str], None] = Field(default_factory=list)
    minimum: Union[list[str], None] = Field(default_factory=list)
    mean: Union[list[str], None] = Field(default_factory=list)
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be imputed."


class ConstantImputerSchema(RootModel[dict[Any, Any]]):
    """Schema for ConstantImputer.

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

            # Accept numeric keys as int or float
            elif isinstance(k, (int, float)):
                if not isinstance(v, Iterable) or isinstance(v, (str, bytes)):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)
                if not all(isinstance(i, str) for i in v):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)

            # Reject all other keys
            else:
                msg = f'Key "{k}" must be either a number, "none" or "default", got {type(k).__name__}'
                raise TypeError(msg)

        return values


class PostprocessorSchema(BaseModel):
    default: str = Field(literals=["none", "relu", "hardtanh", "hardtanh_0_1"])
    "Postprocessor default method to apply."
    relu: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with relu."
    hardtanh: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with hardtanh."
    hardtanh_0_1: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with hardtanh in range [0, 1]."
    none: Union[list[str], None] = Field(default_factory=list)
    "Variables not to be postprocessed."


class NormalizedReluPostprocessorSchema(RootModel[dict[Any, Any]]):
    """Schema for the NormalizedReluPostProcessor.

    Expects the config to have keys corresponding to customizable thresholds and lists of variables
    to postprocess and a normalizer to apply to thresholds.:
    ```
    normalizer: 'mean-std'
    1:
        - y
    0:
        - x
    3.14:
        - q
    ```
    """

    @field_validator("root")
    @classmethod
    def validate_entries(cls, values: dict[Union[int, float, str], Union[str, list[str]]]) -> dict[Any, Any]:

        for k, v in values.items():

            if k == "normalizer":
                if not isinstance(v, str):  #
                    msg = f'"normalizer" must map to a string, got {v}'
                    raise TypeError(msg)
                if v not in ["none", "mean-std", "std", "min-max", "max"]:
                    msg = f'"normalizer" must be one of "none", "mean-std", "std", "min-max", "max", got {v}'
                    raise ValueError(msg)

            # Accept numeric keys as int or float
            elif isinstance(k, (int, float)):
                if not isinstance(v, Iterable) or isinstance(v, (str, bytes)):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)
                if not all(isinstance(i, str) for i in v):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)

            # Reject all other keys
            else:
                msg = f'Key "{k}" must be either a number, "normalizer", got {type(k).__name__}'
                raise TypeError(msg)

        return values


class ConditionalZeroPostprocessorSchema(RootModel[dict[Any, Any]]):
    """Schema for ConditionalZeroPostProcessor.

    Expects the config to have keys corresponding to customizable values and lists of variables
    to postprocess and a variable to use for postprocessing.:

    ```
    default: "none"
    remap: "x"
    0:
        - y
    5.0:
        - x
    3.14:
        - q
    ```

    If "x" is zero, "y" will be postprocessed with 0, "x" with 5.0 and "q" with 3.14.
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
            elif k == "remap":
                if not isinstance(v, str):
                    msg = f'"remap" must map to a strings, got {v}'
                    raise TypeError(msg)

            # Accept numeric keys as int or float
            elif isinstance(k, (int, float)):
                if not isinstance(v, Iterable) or isinstance(v, (str, bytes)):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)
                if not all(isinstance(i, str) for i in v):
                    msg = f'Key "{k}" must map to a list of strings, got {v}'
                    raise TypeError(msg)

            # Reject all other keys
            else:
                msg = f'Key "{k}" must be either a number, "none" or "default", got {type(k).__name__}'
                raise TypeError(msg)

        return values


class ConditionalNaNPostprocessorSchema(BaseModel):
    default: str = Field(literals=["none", "nan"], default="none")
    "Postprocessor default method to apply."
    remap: str
    "Name of conditional variable."
    nan: Union[list[str], None] = Field(default_factory=list)
    "Variables to postprocess with NaNs."
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
    conditional_nan_postprocessor = "anemoi.models.preprocessing.postprocessor.ConditionalNaNPostprocessor"
    normalized_relu_postprocessor = "anemoi.models.preprocessing.postprocessor.NormalizedReluPostprocessor"


target_to_schema = {
    PreprocessorTarget.normalizer: NormalizerSchema,
    PreprocessorTarget.imputer: ImputerSchema,
    PreprocessorTarget.const_imputer: ConstantImputerSchema,
    PreprocessorTarget.remapper: RemapperSchema,
    PreprocessorTarget.postprocessor: PostprocessorSchema,
    PreprocessorTarget.conditional_zero_postprocessor: ConditionalZeroPostprocessorSchema,
    PreprocessorTarget.conditional_nan_postprocessor: ConditionalNaNPostprocessorSchema,
    PreprocessorTarget.normalized_relu_postprocessor: NormalizedReluPostprocessorSchema,
}


class PreprocessorSchema(BaseModel, validate_assignment=False):
    target_: PreprocessorTarget = Field(..., alias="_target_")
    "Processor object from anemoi.models.preprocessing.[normalizer|imputer|remapper]."
    config: Union[dict, NormalizerSchema, ImputerSchema, PostprocessorSchema, RemapperSchema]
    "Target schema containing processor methods."

    @model_validator(mode="after")
    def schema_consistent_with_target(self) -> type["PreprocessorSchema"]:
        schema_cls = target_to_schema.get(self.target_)
        if schema_cls is None:
            error_msg = f"Unknown target: {self.target_}"
            raise ValidationError(error_msg)

        validated = TypeAdapter(schema_cls).validate_python(self.config)
        # If it's a RootModel (like ConstantImputerSchema), extract the root dict
        if hasattr(validated, "root"):
            self.config = validated.root

        return self

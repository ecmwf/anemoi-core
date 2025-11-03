# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from anemoi.models.schemas.data_processor import PreprocessorSchema  # noqa: TC002


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
    forcing : list[str]
        The list of features used as forcing to generate the forecast state.
    diagnostic : list[str]
        The list of features that are only part of the forecast state.
    processors : Dict[str, Processor]
        The Processors configuration.
    target : list[str], optional
        The list of features used to compute the loss against forecasted variables.
    num_features : int, optional
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
    forcing: list[str] = Field(default_factory=list)
    "Features that are not part of the forecast state but are used as forcing to generate the forecast state."
    diagnostic: list[str] = Field(default_factory=list)
    "Features that are only part of the forecast state and are not used as an input to the model."
    target: list[str] | None = None
    (
        "Features used to compute the loss against forecasted variables. "
        "Cannot be prognostic or diagnostic, can have the same name as forcing variables "
        "but have a different role. Such that: prognostic = diagnostic - forcing.union(target)."
    )
    num_features: int | None
    "Number of features in the forecast state. To be set in the code."

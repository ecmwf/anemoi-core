# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class DataModuleSchema(PydanticBaseModel):
    target_: Literal[
        "anemoi.training.data.datamodule.AnemoiEnsDatasetsDataModule",
        "anemoi.training.data.datamodule.AnemoiDatasetsDataModule",
    ] = Field(..., alias="_target_")

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from enum import Enum

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class DefinedAggregatorMethods(str, Enum):
    SUM = "anemoi.models.layers.aggregator.SumAggregator"
    MEAN = "anemoi.models.layers.aggregator.MeanAggregator"
    CONCAT = "anemoi.models.layers.aggregator.ConcatAggregator"


class AggregatorSchema(BaseModel):
    target_: DefinedAggregatorMethods = Field(..., alias="_target_")
    "Aggregator object from anemoi.models.layers.aggregator."

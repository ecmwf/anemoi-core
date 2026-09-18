# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from typing import Literal

import torch

VerticalCoordinateType = Literal[
    "surface",
    "pressure",
    "height",
    "model",
    "layer",
]

TemporalOperator = Literal[
    "instantaneous",
    "accumulation",
    "mean",
    "maximum",
    "minimum",
]


@dataclass(frozen=True)
class VerticalCoordinate:
    type: VerticalCoordinateType  # pressure | surface | model level | height
    level: int | float
    unit: str  # Optional -> hpa | meters | unit-less (model-level)


@dataclass(frozen=True)
class VariableSpecification:
    name: str  # q_100
    param: str  # q -> humidity
    vertical_coordinate: VerticalCoordinate
    temporal_operator: TemporalOperator  # instantenous | accum | mean | max | min
    temporal_window: int | None = None  # i.e 21600
    value_unit: str | None = None  # K


@dataclass(frozen=True)
class DomainVariableMetadata:
    # categorical
    param_ids: torch.Tensor
    vertical_type_ids: torch.Tensor
    temporal_operator_id: torch.Tensor

    # continous
    vertical_levels: torch.Tensor
    temporal_window: torch.Tensor

    # missing value mask
    has_temporal_window: torch.Tensor
    has_vertical_level: torch.Tensor


class VariableVocabular:
    def __init__(self):
        pass


if __name__ == "__main__":
    pass

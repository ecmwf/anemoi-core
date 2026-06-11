# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from anemoi.models.data.source_view import SourceView


@dataclass
class TrainingStepOutput:
    """Output of a training or validation step."""

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    predictions: list[dict[str, SourceView]]
    plot_kwargs: dict[str, Any] = field(default_factory=dict)

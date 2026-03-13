# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training state dataclass for checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class TrainingState:
    """Encapsulates training state extracted from a checkpoint.

    Parameters
    ----------
    epoch : int
        Current training epoch (default: 0)
    global_step : int
        Global training step count (default: 0)
    best_metric : float | None
        Best validation metric seen so far (default: None)
    metrics_history : dict
        History of tracked metrics (default: empty dict)
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    metrics_history: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, checkpoint_data: dict[str, Any]) -> TrainingState:
        """Extract training state from a checkpoint dictionary.

        Gracefully handles missing keys by defaulting to 0.

        Parameters
        ----------
        checkpoint_data : dict
            Checkpoint dictionary (e.g. Lightning-format checkpoint)

        Returns
        -------
        TrainingState
            Extracted training state
        """
        return cls(
            epoch=checkpoint_data.get("epoch", 0),
            global_step=checkpoint_data.get("global_step", 0),
            best_metric=checkpoint_data.get("best_metric"),
            metrics_history=checkpoint_data.get("metrics_history", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns
        -------
        dict
            Dictionary representation of the training state
        """
        return asdict(self)

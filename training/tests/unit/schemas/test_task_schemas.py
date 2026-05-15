# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from anemoi.training.schemas.tasks import ForecasterSchema


def _forecaster_config() -> dict:
    return {
        "_target_": "anemoi.training.tasks.Forecaster",
        "multistep_input": 1,
        "multistep_output": 1,
        "timestep": "5m",
        "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
        "validation_rollout": 1,
    }


def test_forecaster_schema_rejects_legacy_sparse_aliases() -> None:
    with pytest.raises(ValidationError):
        ForecasterSchema(
            **(_forecaster_config() | {"dataset_time_indices": {"radar": {"input": [0], "target": ["5m", "10m"]}}}),
        )

# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
import logging

import pytest
from pydantic import ValidationError

from anemoi.training.schemas.diagnostics import CheckpointSchema
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.hyperparameters import HYPERPARAMETERS_FOR_INFERENCE_CONVERSION
from anemoi.training.utils.hyperparameters import HYPERPARAMETERS_FOR_RESUME
from anemoi.training.utils.hyperparameters import INIT_HYPERPARAMETERS


def test_init_hyperparameters_match_training_module_signature() -> None:
    init_args = set(inspect.signature(BaseTrainingModule.__init__).parameters) - {"self"}
    assert init_args == INIT_HYPERPARAMETERS


def test_required_sets_are_known_hyperparameters() -> None:
    assert HYPERPARAMETERS_FOR_RESUME <= INIT_HYPERPARAMETERS
    assert HYPERPARAMETERS_FOR_INFERENCE_CONVERSION <= INIT_HYPERPARAMETERS


def test_default_excludes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        schema = CheckpointSchema()
    assert schema.exclude_hyperparameters == []
    assert not caplog.records


@pytest.mark.parametrize("name", sorted(HYPERPARAMETERS_FOR_RESUME))
def test_excluding_resume_hyperparameter_raises(name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        CheckpointSchema(exclude_hyperparameters=[name])


def test_excluding_unknown_hyperparameter_raises() -> None:
    with pytest.raises(ValidationError, match="graph_dat"):
        CheckpointSchema(exclude_hyperparameters=["graph_dat"])


def test_excluding_inference_conversion_hyperparameter_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        schema = CheckpointSchema(exclude_hyperparameters=["graph_data", "statistics"])
    assert schema.exclude_hyperparameters == ["graph_data", "statistics"]
    assert "checkpoint inference" in caplog.text
    assert "graph_data" in caplog.text

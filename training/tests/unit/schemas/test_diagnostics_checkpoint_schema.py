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
from argparse import ArgumentParser

import pytest
from pydantic import ValidationError

from anemoi.training.commands.checkpoint import Checkpoint
from anemoi.training.schemas.diagnostics import CheckpointSchema
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.hyperparameters import HYPERPARAMETERS_FOR_RESUME
from anemoi.training.utils.hyperparameters import INIT_HYPERPARAMETERS
from anemoi.training.utils.hyperparameters import OPTIONAL_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION
from anemoi.training.utils.hyperparameters import REQUIRED_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION


def test_init_hyperparameters_match_training_module_signature() -> None:
    init_args = set(inspect.signature(BaseTrainingModule.__init__).parameters) - {"self"}
    assert init_args == INIT_HYPERPARAMETERS


def test_hyperparameter_sets_are_known_hyperparameters() -> None:
    assert HYPERPARAMETERS_FOR_RESUME <= INIT_HYPERPARAMETERS
    assert set(OPTIONAL_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION) <= INIT_HYPERPARAMETERS
    assert REQUIRED_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION <= INIT_HYPERPARAMETERS


def test_inference_conversion_sets_partition_init_hyperparameters() -> None:
    optional = set(OPTIONAL_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION)
    assert not optional & REQUIRED_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION
    assert optional | REQUIRED_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION == INIT_HYPERPARAMETERS


@pytest.mark.parametrize("option", sorted(OPTIONAL_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION.values()))
def test_optional_hyperparameter_options_exist_in_inference_command(option: str) -> None:
    parser = ArgumentParser()
    Checkpoint.add_arguments(parser)
    # fails with SystemExit if the option is not registered
    parser.parse_args(["inference", "-i", "in.ckpt", "-o", "out.ckpt", option, "value"])


def test_default_excludes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        schema = CheckpointSchema()
    assert schema.exclude_hyperparameters == []
    assert not caplog.records


@pytest.mark.parametrize("name", sorted(HYPERPARAMETERS_FOR_RESUME))
def test_excluding_resume_hyperparameter_raises(name: str) -> None:
    with pytest.raises(ValidationError, match="needed to resume training"):
        CheckpointSchema(exclude_hyperparameters=[name])


@pytest.mark.parametrize("name", sorted(REQUIRED_HYPERPARAMETERS_FOR_INFERENCE_CONVERSION - HYPERPARAMETERS_FOR_RESUME))
def test_excluding_conversion_hyperparameter_raises(name: str) -> None:
    with pytest.raises(ValidationError, match="needed by `anemoi-training checkpoint inference`"):
        CheckpointSchema(exclude_hyperparameters=[name])


def test_resume_error_comes_before_conversion_error() -> None:
    # `config` is required by both; `statistics` only by the conversion
    with pytest.raises(ValidationError, match="needed to resume training"):
        CheckpointSchema(exclude_hyperparameters=["statistics", "config"])


def test_excluding_unknown_hyperparameter_raises() -> None:
    with pytest.raises(ValidationError, match="graph_dat"):
        CheckpointSchema(exclude_hyperparameters=["graph_dat"])

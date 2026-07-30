# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for CheckpointSource base class."""

import pytest

from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.sources.base import CheckpointSource


def test_checkpoint_source_extends_pipeline_stage() -> None:
    """CheckpointSource MUST inherit from PipelineStage."""
    assert issubclass(CheckpointSource, PipelineStage)


def test_checkpoint_source_is_abstract() -> None:
    """Cannot instantiate CheckpointSource directly."""
    with pytest.raises(TypeError):
        CheckpointSource()


def test_checkpoint_source_requires_process() -> None:
    """Subclasses must implement async process(context) -> context."""
    import inspect

    method = CheckpointSource.process
    assert inspect.iscoroutinefunction(method) or hasattr(method, "__isabstractmethod__")

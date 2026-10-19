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

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.sources.base import CheckpointSource


def test_checkpoint_source_extends_pipeline_stage() -> None:
    """CheckpointSource MUST inherit from PipelineStage."""
    assert issubclass(CheckpointSource, PipelineStage)


class _ProcessOnlySource(CheckpointSource):
    """Source with ``process`` only, no ``resolve``."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


def test_resolve_is_not_required_for_process_only_sources_but_names_them_on_resume() -> None:
    """A ``process``-only source loads, but ``resolve`` raises with the class name."""
    source = _ProcessOnlySource()
    context = CheckpointContext()

    assert source.process(context) is context
    with pytest.raises(NotImplementedError, match=r"_ProcessOnlySource.*resolve"):
        source.resolve(context)
    assert not _ProcessOnlySource.supports_resume()


def test_checkpoint_source_is_abstract() -> None:
    """Cannot instantiate CheckpointSource directly."""
    with pytest.raises(TypeError):
        CheckpointSource()


def test_checkpoint_source_requires_process() -> None:
    """Subclasses must implement process(context) -> context."""
    import inspect

    method = CheckpointSource.process
    assert inspect.iscoroutinefunction(method) or hasattr(method, "__isabstractmethod__")

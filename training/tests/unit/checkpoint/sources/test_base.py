# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for CheckpointSource base class."""

from pathlib import Path

import pytest

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.base import remove_temporary_file


def test_checkpoint_source_extends_pipeline_stage() -> None:
    """CheckpointSource MUST inherit from PipelineStage."""
    assert issubclass(CheckpointSource, PipelineStage)


class _ProcessOnlySource(CheckpointSource):
    """A source written against the old contract: ``process`` only, no ``resolve``."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


@pytest.mark.asyncio
async def test_resolve_is_not_required_for_process_only_sources_but_names_them_on_resume() -> None:
    """A ``process``-only source still works for pipeline loads; a resume needs ``resolve``.

    The error names the class so the author knows what to implement.
    """
    source = _ProcessOnlySource()
    context = CheckpointContext()

    assert await source.process(context) is context
    with pytest.raises(NotImplementedError, match=r"_ProcessOnlySource.*resolve"):
        await source.resolve(context)


def test_remove_temporary_file_tolerates_a_file_that_is_already_gone(tmp_path: Path) -> None:
    """The trainer's cleanup and the atexit backstop may both run; the second is a no-op."""
    path = tmp_path / "download.ckpt"
    path.write_bytes(b"x")

    remove_temporary_file(path)
    assert not path.exists()
    remove_temporary_file(path)  # must not raise


def test_checkpoint_source_is_abstract() -> None:
    """Cannot instantiate CheckpointSource directly."""
    with pytest.raises(TypeError):
        CheckpointSource()


def test_checkpoint_source_requires_process() -> None:
    """Subclasses must implement async process(context) -> context."""
    import inspect

    method = CheckpointSource.process
    assert inspect.iscoroutinefunction(method) or hasattr(method, "__isabstractmethod__")

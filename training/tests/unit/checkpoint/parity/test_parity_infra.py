# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity coverage for the checkpoint infrastructure and contract layer.

These tests close genuine assertion gaps in the shared checkpoint
infrastructure: the :class:`CheckpointContext` contract
(``validate_for_stage`` and ``__repr__``), the format-availability helper,
the ``download_with_retry`` error-mapping paths, the ``CheckpointPipeline``
orchestration branches (Hydra ``add_stage`` instantiation, running-loop
``execute_sync``, ``MemoryError`` propagation, completion markers, and the
validation-import fallback), the post-run ``validate_pipeline_health``
invariants, the ``ComponentCatalog`` discovery cache and suggestion paths,
and the checkpoint exception message builders.

Each test drives the real code path on small CPU objects with deterministic
inputs and asserts the concrete observable signal (raised exception type,
exact message substrings, metadata values, call counts) rather than a weaker
proxy.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import aiohttp
import pytest
import torch.nn as nn

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.catalog import ComponentCatalog
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointError
from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.exceptions import CheckpointSourceError
from anemoi.training.checkpoint.exceptions import CheckpointTimeoutError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.formats import is_format_available
from anemoi.training.checkpoint.pipeline import CheckpointPipeline
from anemoi.training.checkpoint.sources import LocalSource
from anemoi.training.checkpoint.utils import download_with_retry
from anemoi.training.checkpoint.validation import validate_pipeline_health

if TYPE_CHECKING:
    from pathlib import Path


class MockStage(PipelineStage):
    """Minimal successful stage that records a per-name marker in metadata."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.process_called = False

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        self.process_called = True
        context.update_metadata(**{f"stage_{self.name}": "processed"})
        return context


class _MemoryErrorStage(PipelineStage):
    """Stage whose process raises MemoryError."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:  # noqa: ARG002
        msg = "out of memory"
        raise MemoryError(msg)


class _CheckpointErrorStage(PipelineStage):
    """Stage whose process raises a plain CheckpointError."""

    async def process(self, context: CheckpointContext) -> CheckpointContext:  # noqa: ARG002
        msg = "stage boom"
        raise CheckpointError(msg)


# ---------------------------------------------------------------------------
# CheckpointContext contract
# ---------------------------------------------------------------------------


def test_validate_for_stage_reports_missing_required_fields() -> None:
    """validate_for_stage raises CheckpointError naming the missing fields."""
    empty = CheckpointContext()
    with pytest.raises(CheckpointError) as exc:
        empty.validate_for_stage("LoadingStage", ["model", "checkpoint_data"])
    message = str(exc.value)
    assert "model" in message
    assert "checkpoint_data" in message

    # An empty (falsy) checkpoint_data counts as missing via has_checkpoint_data().
    empty_data = CheckpointContext(model=nn.Linear(2, 2), checkpoint_data={})
    with pytest.raises(CheckpointError) as exc_empty:
        empty_data.validate_for_stage("LoadingStage", ["model", "checkpoint_data"])
    empty_message = str(exc_empty.value)
    assert "checkpoint_data" in empty_message
    # 'model' is populated, so it must not appear in the missing-field list.
    assert "'model'" not in empty_message

    # A fully populated context does not raise.
    populated = CheckpointContext(model=nn.Linear(2, 2), checkpoint_data={"state_dict": {}})
    populated.validate_for_stage("LoadingStage", ["model", "checkpoint_data"])


def test_context_repr_omits_unset_fields(tmp_path: Path) -> None:
    """__repr__ renders only the fields that are set."""
    assert repr(CheckpointContext()) == "CheckpointContext()"

    metadata_only = repr(CheckpointContext(metadata={"epoch": 5}))
    assert "metadata_keys=['epoch']" in metadata_only
    assert "path=" not in metadata_only
    assert "model=" not in metadata_only

    model_only = repr(CheckpointContext(model=nn.Linear(2, 2)))
    assert "model=Linear" in model_only
    assert "path=" not in model_only
    assert "metadata_keys" not in model_only

    all_set = repr(
        CheckpointContext(
            checkpoint_path=tmp_path / "model.ckpt",
            model=nn.Linear(2, 2),
            metadata={"a": 1},
        ),
    )
    assert "path=model.ckpt" in all_set
    assert "model=Linear" in all_set
    assert "metadata_keys=['a']" in all_set


# ---------------------------------------------------------------------------
# Format availability
# ---------------------------------------------------------------------------


def test_is_format_available_false_for_unsupported_formats() -> None:
    """is_format_available returns False for unsupported/unknown formats."""
    assert is_format_available("safetensors") is False
    assert is_format_available("unknown") is False

    assert is_format_available("lightning") is True
    assert is_format_available("pytorch") is True
    assert is_format_available("state_dict") is True


# ---------------------------------------------------------------------------
# download_with_retry error mapping
# ---------------------------------------------------------------------------


def _session_raising_on_enter(error: Exception) -> AsyncMock:
    """Build a mocked ClientSession whose session.get context enter raises ``error``."""
    get_context = AsyncMock()
    get_context.__aenter__ = AsyncMock(side_effect=error)
    get_context.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.get = Mock(return_value=get_context)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _session_raising_on_status(error: Exception) -> AsyncMock:
    """Build a mocked ClientSession whose response.raise_for_status raises ``error``."""
    response = Mock()
    response.raise_for_status = Mock(side_effect=error)
    response.headers = {"content-length": "0"}

    get_context = AsyncMock()
    get_context.__aenter__ = AsyncMock(return_value=response)
    get_context.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.get = Mock(return_value=get_context)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def test_download_timeout_maps_to_checkpoint_timeout_error(tmp_path: Path) -> None:
    """A timeout on the final attempt maps to CheckpointTimeoutError."""
    session = _session_raising_on_enter(TimeoutError("timed out"))
    dest = tmp_path / "model.ckpt"

    with patch("aiohttp.ClientSession", return_value=session), pytest.raises(CheckpointTimeoutError):
        await download_with_retry("https://example.com/model.ckpt", dest, max_retries=1, timeout=5)


async def test_client_error_4xx_no_retry_but_5xx_retries(tmp_path: Path) -> None:
    """4xx raises immediately (1 attempt); 5xx retries before raising."""
    err_404 = aiohttp.ClientResponseError(
        request_info=Mock(real_url="https://example.com/a.ckpt"),
        history=(),
        status=404,
        message="Not Found",
    )
    session_404 = _session_raising_on_status(err_404)
    with (
        patch("aiohttp.ClientSession", return_value=session_404),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(CheckpointSourceError),
    ):
        await download_with_retry("https://example.com/a.ckpt", tmp_path / "a.ckpt", max_retries=3, timeout=5)
    assert session_404.get.call_count == 1

    err_503 = aiohttp.ClientResponseError(
        request_info=Mock(real_url="https://example.com/b.ckpt"),
        history=(),
        status=503,
        message="Service Unavailable",
    )
    session_503 = _session_raising_on_status(err_503)
    with (
        patch("aiohttp.ClientSession", return_value=session_503),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(CheckpointSourceError),
    ):
        await download_with_retry("https://example.com/b.ckpt", tmp_path / "b.ckpt", max_retries=2, timeout=5)
    assert session_503.get.call_count == 2

    assert session_503.get.call_count > session_404.get.call_count


# ---------------------------------------------------------------------------
# CheckpointPipeline orchestration
# ---------------------------------------------------------------------------


def test_add_stage_instantiates_config_and_rejects_invalid(tmp_path: Path) -> None:
    """add_stage instantiates dict/_target_ configs and rejects invalid stages."""
    pipeline = CheckpointPipeline([])
    pipeline.add_stage(
        {
            "_target_": "anemoi.training.checkpoint.sources.LocalSource",
            "path": str(tmp_path / "model.ckpt"),
        },
    )
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], LocalSource)

    # A plain object has no process method -> rejected.
    with pytest.raises(CheckpointConfigError):
        CheckpointPipeline([]).add_stage(object())

    # An unimportable _target_ fails Hydra instantiation -> rejected.
    with pytest.raises(CheckpointConfigError):
        CheckpointPipeline([]).add_stage({"_target_": "nonexistent.module.DoesNotExist"})


async def test_execute_sync_uses_thread_pool_inside_running_loop() -> None:
    """execute_sync works when called from within a running event loop."""
    stage1 = MockStage("a")
    stage2 = MockStage("b")
    pipeline = CheckpointPipeline([stage1, stage2])

    # This test function runs inside a live event loop; execute_sync must
    # detect it and offload to a thread pool rather than raising.
    result = pipeline.execute_sync(CheckpointContext(model=nn.Linear(2, 2)))

    assert stage1.process_called
    assert stage2.process_called
    assert result.metadata["stage_0_MockStage"] == "completed"
    assert result.metadata["stage_1_MockStage"] == "completed"


async def test_memory_error_reraised_even_with_continue_on_error() -> None:
    """MemoryError propagates but CheckpointError is swallowed under continue_on_error."""
    memory_pipeline = CheckpointPipeline([_MemoryErrorStage()], continue_on_error=True)
    with pytest.raises(MemoryError):
        await memory_pipeline.execute_async(CheckpointContext(model=nn.Linear(2, 2)))

    error_pipeline = CheckpointPipeline([_CheckpointErrorStage()], continue_on_error=True)
    result = await error_pipeline.execute_async(CheckpointContext(model=nn.Linear(2, 2)))
    stage_values = [v for k, v in result.metadata.items() if k.startswith("stage_")]
    assert any("failed" in str(v) for v in stage_values)


async def test_completed_markers_recorded_for_successful_stages() -> None:
    """Successful stages record 'completed' markers in metadata."""
    pipeline = CheckpointPipeline([MockStage("a"), MockStage("b")])
    result = await pipeline.execute_async(CheckpointContext(model=nn.Linear(2, 2)))

    assert result.metadata["stage_0_MockStage"] == "completed"
    assert result.metadata["stage_1_MockStage"] == "completed"


async def test_pre_execution_validation_degrades_on_import_error() -> None:
    """Pipeline degrades gracefully when the validation module import fails."""
    pipeline = CheckpointPipeline([MockStage("only")])
    context = CheckpointContext(model=nn.Linear(2, 2))

    # Setting the module entry to None forces the local
    # `from .validation import CheckpointPipelineValidator` to raise ImportError.
    with patch.dict(sys.modules, {"anemoi.training.checkpoint.validation": None}):
        result = await pipeline.execute(context)

    assert result.metadata["validation_performed"] is False
    assert result.metadata["validation_skipped"] == "module_unavailable"


# ---------------------------------------------------------------------------
# validate_pipeline_health invariants
# ---------------------------------------------------------------------------


def test_scheduler_without_optimizer_is_flagged() -> None:
    """A scheduler present without an optimizer is flagged."""
    context = CheckpointContext(model=nn.Linear(2, 2), optimizer=None, scheduler=object())
    context.update_metadata(stage_0_Loading="completed")

    with pytest.raises(CheckpointValidationError) as exc:
        validate_pipeline_health(context)

    assert "Scheduler present but optimizer is None" in exc.value.validation_errors


def test_non_string_stage_value_is_flagged() -> None:
    """A stage_N_ metadata entry with a non-string value is flagged."""
    context = CheckpointContext(model=nn.Linear(2, 2))
    context.metadata["stage_0_X"] = 123

    with pytest.raises(CheckpointValidationError) as exc:
        validate_pipeline_health(context)

    assert any("Stage entry 'stage_0_X' has non-string value" in err for err in exc.value.validation_errors)


def test_config_validation_error_status_is_propagated() -> None:
    """A configuration-validation 'error' status propagates to a health failure."""
    context = CheckpointContext(model=nn.Linear(2, 2))
    context.update_metadata(stage_0_Loading="completed", validation_config_status="error")

    with pytest.raises(CheckpointValidationError) as exc:
        validate_pipeline_health(context)

    assert any("configuration validation" in err for err in exc.value.validation_errors)


# ---------------------------------------------------------------------------
# ComponentCatalog discovery + suggestions
# ---------------------------------------------------------------------------


class TestComponentCatalogParity:
    """Catalog tests that touch the class-level discovery cache."""

    @pytest.fixture(autouse=True)
    def _reset_catalog_cache(self) -> None:
        ComponentCatalog._sources = None
        ComponentCatalog._loaders = None
        ComponentCatalog._modifiers = None
        yield
        ComponentCatalog._sources = None
        ComponentCatalog._loaders = None
        ComponentCatalog._modifiers = None

    def test_discovery_result_is_cached(self) -> None:
        """list_* discovery runs once and the result is cached."""
        fixed = {
            "s3": "anemoi.training.checkpoint.sources.S3Source",
            "local": "anemoi.training.checkpoint.sources.LocalSource",
        }
        with patch.object(ComponentCatalog, "_discover_components", return_value=dict(fixed)) as mock_discover:
            ComponentCatalog._sources = None
            first = ComponentCatalog.list_sources()
            second = ComponentCatalog.list_sources()

        assert mock_discover.call_count == 1
        assert first == second == ["local", "s3"]

    def test_get_source_target_unknown_lists_available_and_suggests(self) -> None:
        """Unknown source name against a populated catalog lists names + suggestions."""
        fixed = {
            "local": "anemoi.training.checkpoint.sources.LocalSource",
            "http": "anemoi.training.checkpoint.sources.HTTPSource",
        }
        with patch.object(ComponentCatalog, "_discover_components", return_value=dict(fixed)):
            ComponentCatalog._sources = None

            # The available-names listing is emitted for any unknown name.
            with pytest.raises(CheckpointConfigError) as exc:
                ComponentCatalog.get_source_target("locel")
            message = str(exc.value)
            assert "Available sources:" in message
            assert "local" in message
            assert "http" in message

            # The similar-name suggestion branch fires for a substring near-miss.
            with pytest.raises(CheckpointConfigError) as exc_suggest:
                ComponentCatalog.get_source_target("loca")
            assert "Did you mean: local" in str(exc_suggest.value)

    def test_find_similar_names_matches_substring(self) -> None:
        """_find_similar_names returns substring near-matches."""
        result = ComponentCatalog._find_similar_names("weight", ["weights_only", "warm_start"])
        assert "weights_only" in result
        assert "warm_start" not in result

        # Reached end-to-end via get_loader_target's error path on a populated catalog.
        fixed = {
            "weights_only": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader",
            "warm_start": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader",
        }
        with patch.object(ComponentCatalog, "_discover_components", return_value=dict(fixed)):
            ComponentCatalog._loaders = None
            with pytest.raises(CheckpointConfigError) as exc:
                ComponentCatalog.get_loader_target("weight")
        assert "Did you mean: weights_only" in str(exc.value)


# ---------------------------------------------------------------------------
# Exception message builders
# ---------------------------------------------------------------------------


def test_not_found_error_builds_suggestions(tmp_path: Path) -> None:
    """CheckpointNotFoundError builds similar-files / missing-directory hints."""
    (tmp_path / "sibling.ckpt").write_bytes(b"x")
    (tmp_path / "other.pt").write_bytes(b"y")

    similar = CheckpointNotFoundError(tmp_path / "missing.ckpt")
    assert "Found similar files" in str(similar)

    missing_dir = CheckpointNotFoundError(tmp_path / "nonexistent_dir" / "model.ckpt")
    assert "Directory does not exist" in str(missing_dir)


def test_incompatible_error_truncates_key_lists() -> None:
    """CheckpointIncompatibleError truncates key lists to 5 + '... and N more'."""
    missing_keys = [f"layer.{i}.weight" for i in range(6)]
    missing_error = CheckpointIncompatibleError("Mismatch", missing_keys=missing_keys)
    missing_text = str(missing_error)
    assert "... and 1 more" in missing_text
    assert missing_keys[5] not in missing_text

    unexpected_keys = [f"extra.{i}" for i in range(6)]
    unexpected_error = CheckpointIncompatibleError("Mismatch", unexpected_keys=unexpected_keys)
    unexpected_text = str(unexpected_error)
    assert "... and 1 more" in unexpected_text
    assert unexpected_keys[5] not in unexpected_text

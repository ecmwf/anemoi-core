# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity coverage for the checkpoint acquisition (sources) subsystem.

These tests close verification gaps in the source layer that no other suite
exercises: LocalSource path precedence / tilde-expansion / symlink resolution,
the ``CheckpointNotFoundError`` suggestion branches, the torch.load error-type
wrapping into ``CheckpointLoadError``, the ``weights_only=False`` guarantee,
RunSource rank-detection fall-throughs (LOCAL_RANK / SLURM_PROCID /
JSM_NAMESPACE_RANK / malformed / negative), the auxiliary-key Lightning branch
in ``detect_format_from_data``, HTTPSource construction/validation and checksum
handling, and the S3 URL-parsing / ``supports`` / URL-resolution pure logic.

Everything runs on CPU with no network. Remote downloads are replaced by a
synthetic writer that copies a real on-disk checkpoint into the destination the
source chose, so the source code under test always runs end to end.
"""

from __future__ import annotations

import argparse
import inspect
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.exceptions import CheckpointNotFoundError
from anemoi.training.checkpoint.exceptions import CheckpointValidationError
from anemoi.training.checkpoint.formats import detect_format_from_data
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.http import HTTPSource
from anemoi.training.checkpoint.sources.local import LocalSource
from anemoi.training.checkpoint.sources.run import RunSource
from anemoi.training.checkpoint.sources.s3 import S3Source
from anemoi.training.checkpoint.utils import calculate_checksum

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from collections.abc import Callable
    from collections.abc import Iterable

_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


def _write_state_dict_ckpt(path: Path) -> Path:
    """Write a minimal Lightning-shaped checkpoint to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"layer.weight": torch.zeros(2, 2)}}, path)
    return path


def _run_config(root: Path) -> OmegaConf:
    return OmegaConf.create({"system": {"output": {"checkpoints": {"root": str(root)}}}})


def _clear_rank_env(monkeypatch: pytest.MonkeyPatch, keep: Iterable[str] = ()) -> None:
    for var in _RANK_ENV_VARS:
        if var not in keep:
            monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# LocalSource path handling
# ---------------------------------------------------------------------------


async def test_local_source_explicit_path_overrides_context_checkpoint_path(tmp_path: Path) -> None:
    """An explicit ``path=`` wins over an already-set ``context.checkpoint_path``."""
    configured = _write_state_dict_ckpt(tmp_path / "configured" / "model.ckpt")
    other = _write_state_dict_ckpt(tmp_path / "other" / "other.ckpt")

    context = CheckpointContext(checkpoint_path=other)
    result = await LocalSource(path=configured).process(context)

    assert result.checkpoint_path == configured
    assert result.metadata["source_path"] == str(configured.resolve())


async def test_local_source_expands_tilde_in_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``~/...`` is expanded via expanduser before loading; the recorded path is absolute."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_state_dict_ckpt(tmp_path / "model.ckpt")

    context = CheckpointContext(checkpoint_path="~/model.ckpt")
    result = await LocalSource().process(context)

    source_path = result.metadata["source_path"]
    assert source_path.startswith("/")
    assert "~" not in source_path
    assert source_path == str((tmp_path / "model.ckpt").resolve())


async def test_local_source_resolves_symlink_to_canonical_path(tmp_path: Path) -> None:
    """A symlinked checkpoint is followed to its canonical target by resolve()."""
    actual = _write_state_dict_ckpt(tmp_path / "actual.ckpt")
    link = tmp_path / "link.ckpt"
    link.symlink_to(actual)

    context = CheckpointContext(checkpoint_path=link)
    result = await LocalSource().process(context)

    assert result.metadata["source_path"] == str(actual.resolve())


async def test_local_source_missing_file_lists_similar_files(tmp_path: Path) -> None:
    """A missing file whose parent holds other checkpoints suggests those files."""
    directory = tmp_path / "ckpts"
    directory.mkdir()
    (directory / "epoch_1.ckpt").write_bytes(b"x")
    (directory / "epoch_2.pt").write_bytes(b"x")

    context = CheckpointContext(checkpoint_path=directory / "missing.ckpt")
    with pytest.raises(CheckpointNotFoundError) as excinfo:
        await LocalSource().process(context)

    assert "similar files" in str(excinfo.value)


async def test_local_source_missing_parent_directory_reports_missing_directory(tmp_path: Path) -> None:
    """A missing parent directory produces the 'Directory does not exist' suggestion."""
    context = CheckpointContext(checkpoint_path=tmp_path / "no_such_dir" / "model.ckpt")
    with pytest.raises(CheckpointNotFoundError) as excinfo:
        await LocalSource().process(context)

    assert "Directory does not exist" in str(excinfo.value)


def _make_directory_ckpt(tmp_path: Path) -> Path:
    target = tmp_path / "a_directory.ckpt"
    target.mkdir()
    return target


def _make_empty_ckpt(tmp_path: Path) -> Path:
    target = tmp_path / "empty.ckpt"
    target.write_bytes(b"")
    return target


def _make_garbage_ckpt(tmp_path: Path) -> Path:
    target = tmp_path / "garbage.ckpt"
    target.write_bytes(b"\x00\x01\x02not a checkpoint\xff\xfe")
    return target


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    [
        (_make_directory_ckpt, OSError),
        (_make_empty_ckpt, EOFError),
        (_make_garbage_ckpt, pickle.UnpicklingError),
    ],
)
async def test_local_source_wraps_load_errors(
    tmp_path: Path,
    factory: Callable[[Path], Path],
    expected_type: type[Exception],
) -> None:
    """torch.load failures are wrapped in CheckpointLoadError, preserving the original error type."""
    bad_path = factory(tmp_path)
    context = CheckpointContext(checkpoint_path=bad_path)

    with pytest.raises(CheckpointLoadError) as excinfo:
        await LocalSource().process(context)

    assert isinstance(excinfo.value.original_error, expected_type)


async def test_local_source_loads_lightning_metadata_requiring_weights_only_false(tmp_path: Path) -> None:
    """LocalSource loads Lightning metadata that ``weights_only=True`` would reject."""
    ckpt = {
        "pytorch-lightning_version": "2.2.0",
        "callbacks": {"cb": {"best": 0.1}},
        "hyper_parameters": {"config": argparse.Namespace(lr=1e-4)},
        "state_dict": {"layer.weight": torch.zeros(2, 2)},
    }
    path = tmp_path / "lightning.ckpt"
    torch.save(ckpt, path)

    # The fixture is only meaningful if weights_only=True genuinely rejects it,
    # otherwise the assertion below could not distinguish the two load modes.
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, weights_only=True, map_location="cpu")

    context = CheckpointContext(checkpoint_path=path)
    result = await LocalSource().process(context)

    assert result.checkpoint_data is not None
    for key in ("callbacks", "hyper_parameters", "pytorch-lightning_version"):
        assert key in result.checkpoint_data
    assert result.checkpoint_format == "lightning"


async def test_local_source_s3_scheme_path_treated_as_local_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``s3://`` string given to LocalSource is a literal local path -> NotFound (no dispatch)."""
    monkeypatch.chdir(tmp_path)
    context = CheckpointContext(checkpoint_path="s3://bucket/model.ckpt")
    with pytest.raises(CheckpointNotFoundError):
        await LocalSource().process(context)


# ---------------------------------------------------------------------------
# LocalSource.supports
# ---------------------------------------------------------------------------


def test_local_supports_path_object_true() -> None:
    assert LocalSource.supports(Path("/models/model.ckpt")) is True


def test_local_supports_existing_file_true(tmp_path: Path) -> None:
    ckpt = _write_state_dict_ckpt(tmp_path / "model.ckpt")
    assert LocalSource.supports(str(ckpt)) is True


def test_local_supports_scheme_urls_false() -> None:
    assert LocalSource.supports("s3://bucket/model.ckpt") is False
    assert LocalSource.supports("http://example.com/model.ckpt") is False
    assert LocalSource.supports("https://example.com/model.ckpt") is False
    assert LocalSource.supports("gs://bucket/model.ckpt") is False


# ---------------------------------------------------------------------------
# CheckpointNotFoundError determinism
# ---------------------------------------------------------------------------


def test_checkpoint_not_found_error_is_deterministic(tmp_path: Path) -> None:
    """Repeated construction on the same missing path yields the same type and message."""
    missing = tmp_path / "no_such_dir" / "model.ckpt"
    first = CheckpointNotFoundError(missing)
    second = CheckpointNotFoundError(missing)

    assert type(first) is type(second)
    assert str(first) == str(second)
    assert first.path == second.path


# ---------------------------------------------------------------------------
# RunSource rank detection fall-through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank_var", ["LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"])
async def test_run_source_defers_on_nonzero_rank_from_fallback_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_var: str,
) -> None:
    """When RANK is unset, a non-zero fallback rank variable defers instead of raising."""
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv(rank_var, "1")

    context = CheckpointContext(config=_run_config(tmp_path / "job" / "checkpoints"))
    result = await RunSource(run_id="run_missing").process(context)

    assert result.checkpoint_path is None
    assert result.checkpoint_data is None


async def test_run_source_rank_precedence_rank_over_local_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RANK is read before LOCAL_RANK: RANK=1 defers even though LOCAL_RANK=0."""
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    context = CheckpointContext(config=_run_config(tmp_path / "job" / "checkpoints"))
    result = await RunSource(run_id="run_missing").process(context)

    assert result.checkpoint_path is None


@pytest.mark.parametrize("rank_value", ["abc", "-1"])
async def test_run_source_invalid_rank_treated_as_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_value: str,
) -> None:
    """A malformed or negative RANK is treated conservatively as rank 0 (raises on missing)."""
    _clear_rank_env(monkeypatch)
    monkeypatch.setenv("RANK", rank_value)

    context = CheckpointContext(config=_run_config(tmp_path / "job" / "checkpoints"))
    with pytest.raises(RuntimeError, match="run_missing"):
        await RunSource(run_id="run_missing").process(context)


async def test_run_source_sets_resolved_checkpoint_path_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved run checkpoint records its full path string in metadata."""
    _clear_rank_env(monkeypatch)
    root = tmp_path / "job" / "checkpoints"
    resolved = _write_state_dict_ckpt(tmp_path / "job" / "run_A" / "last.ckpt")

    context = CheckpointContext(config=_run_config(root))
    result = await RunSource(run_id="run_A").process(context)

    assert isinstance(result.metadata["resolved_checkpoint_path"], str)
    assert result.metadata["resolved_checkpoint_path"] == str(resolved)


# ---------------------------------------------------------------------------
# detect_format_from_data
# ---------------------------------------------------------------------------


def test_detect_format_lightning_from_auxiliary_keys_without_version() -> None:
    """Two Lightning auxiliary keys (no version key) are enough to classify as lightning."""
    data = {"callbacks": {}, "optimizer_states": [{}]}
    assert detect_format_from_data(data) == "lightning"


def test_detect_format_pytorch_from_model_state_dict() -> None:
    data = {"model_state_dict": {"w": torch.zeros(2)}, "epoch": 3}
    assert detect_format_from_data(data) == "pytorch"


def test_detect_format_state_dict_from_raw_tensors() -> None:
    data = {"layer.weight": torch.zeros(2, 2), "layer.bias": torch.zeros(2)}
    assert detect_format_from_data(data) == "state_dict"


def test_detect_format_defaults_to_pytorch_for_unrecognized() -> None:
    data = {"foo": 1, "bar": "baz"}
    assert detect_format_from_data(data) == "pytorch"


def test_detect_format_is_deterministic_across_calls() -> None:
    """Format detection is a pure function of the data: repeated calls agree."""
    data = {"callbacks": {}, "optimizer_states": [{}]}
    first = detect_format_from_data(data)
    second = detect_format_from_data(data)
    assert first == second == "lightning"


async def test_local_source_path_expansion_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading ``~/model.ckpt`` twice resolves to the same absolute path each time."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_state_dict_ckpt(tmp_path / "model.ckpt")

    first = await LocalSource().process(CheckpointContext(checkpoint_path="~/model.ckpt"))
    second = await LocalSource().process(CheckpointContext(checkpoint_path="~/model.ckpt"))

    assert first.metadata["source_path"] == second.metadata["source_path"]


# ---------------------------------------------------------------------------
# CheckpointSource base helper and class contracts
# ---------------------------------------------------------------------------


def test_load_and_populate_mutates_context_in_place() -> None:
    """``_load_and_populate`` attaches the raw dict and sets the detected format in place."""
    raw_data = {"layer.weight": torch.zeros(2, 2)}
    context = CheckpointContext()

    LocalSource()._load_and_populate(context, raw_data)

    assert context.checkpoint_data is raw_data
    assert context.checkpoint_format == "state_dict"


def test_all_sources_subclass_checkpoint_source() -> None:
    for source_cls in (LocalSource, RunSource, HTTPSource, S3Source):
        assert issubclass(source_cls, CheckpointSource)


def test_all_sources_process_is_async() -> None:
    for source_cls in (LocalSource, RunSource, HTTPSource, S3Source):
        assert inspect.iscoroutinefunction(source_cls.process)


def test_supports_is_staticmethod_on_string_dispatched_sources() -> None:
    for source_cls in (LocalSource, HTTPSource, S3Source):
        assert isinstance(source_cls.__dict__["supports"], staticmethod)


# ---------------------------------------------------------------------------
# HTTPSource construction and validation
# ---------------------------------------------------------------------------


def test_http_source_accepts_http_url() -> None:
    source = HTTPSource(url="http://example.com/model.ckpt")
    assert source.url == "http://example.com/model.ckpt"


def test_http_source_accepts_https_url() -> None:
    source = HTTPSource(url="https://example.com/model.ckpt")
    assert source.url == "https://example.com/model.ckpt"


def test_http_source_rejects_ftp_scheme() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        HTTPSource(url="ftp://example.com/model.ckpt")
    message = str(excinfo.value)
    assert "scheme" in message
    assert "ftp" in message


def test_http_source_rejects_s3_scheme() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        HTTPSource(url="s3://bucket/model.ckpt")
    message = str(excinfo.value)
    assert "scheme" in message
    assert "s3" in message


def test_http_source_rejects_url_without_host() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        HTTPSource(url="http:///model.ckpt")
    assert "host" in str(excinfo.value)


def test_http_source_allows_host_without_path() -> None:
    source = HTTPSource(url="http://localhost")
    assert source.url == "http://localhost"


def test_http_source_stores_max_retries() -> None:
    source = HTTPSource(url="https://example.com/model.ckpt", max_retries=5)
    assert source.max_retries == 5


def test_http_source_stores_timeout() -> None:
    source = HTTPSource(url="https://example.com/model.ckpt", timeout=600)
    assert source.timeout == 600


def test_http_source_default_checksum_is_none() -> None:
    source = HTTPSource(url="https://example.com/model.ckpt")
    assert source.expected_checksum is None


def test_http_source_stores_expected_checksum() -> None:
    checksum = "6b86b273f403ebc8370ba1e7c2cc8d6f"
    source = HTTPSource(url="https://example.com/model.ckpt", expected_checksum=checksum)
    assert source.expected_checksum == checksum


def test_http_supports_http_and_https() -> None:
    assert HTTPSource.supports("http://example.com/model.ckpt") is True
    assert HTTPSource.supports("https://example.com/model.ckpt") is True


def test_http_supports_rejects_path_object() -> None:
    assert HTTPSource.supports(Path("/models/model.ckpt")) is False


def test_http_supports_rejects_s3_url() -> None:
    assert HTTPSource.supports("s3://bucket/model.ckpt") is False


def test_http_supports_rejects_local_path() -> None:
    assert HTTPSource.supports("/models/model.ckpt") is False


# ---------------------------------------------------------------------------
# HTTPSource download / checksum handling (download replaced by a local copy)
# ---------------------------------------------------------------------------


def _copy_download_stub(source_file: Path, recorded: list[Path]) -> Callable[..., Awaitable[Path]]:
    """Build an async replacement for ``download_with_retry`` that copies a real file."""

    async def _fake_download(*, url: str, dest: Path, max_retries: int, timeout: int) -> Path:  # noqa: ARG001
        recorded.append(Path(dest))
        shutil.copyfile(source_file, dest)
        return Path(dest)

    return _fake_download


def _garbage_download_stub(recorded: list[Path]) -> Callable[..., Awaitable[Path]]:
    async def _fake_download(*, url: str, dest: Path, max_retries: int, timeout: int) -> Path:  # noqa: ARG001
        recorded.append(Path(dest))
        Path(dest).write_bytes(b"\x00not a checkpoint\xff")
        return Path(dest)

    return _fake_download


async def test_http_source_checksum_match_loads_and_sets_metadata(tmp_path: Path) -> None:
    """A matching checksum loads the checkpoint and records http source metadata."""
    good = _write_state_dict_ckpt(tmp_path / "good.ckpt")
    checksum = calculate_checksum(good)
    recorded: list[Path] = []

    source = HTTPSource(url="https://example.com/model.ckpt", expected_checksum=checksum)
    with patch("anemoi.training.checkpoint.utils.download_with_retry", _copy_download_stub(good, recorded)):
        result = await source.process(CheckpointContext())

    assert recorded, "download stub was not invoked"
    assert result.checkpoint_data is not None
    assert result.metadata["source_type"] == "http"
    assert result.metadata["source_url"] == "https://example.com/model.ckpt"


async def test_http_source_checksum_mismatch_raises_validation_error(tmp_path: Path) -> None:
    good = _write_state_dict_ckpt(tmp_path / "good.ckpt")
    recorded: list[Path] = []

    source = HTTPSource(url="https://example.com/model.ckpt", expected_checksum="deadbeef")
    with (
        patch("anemoi.training.checkpoint.utils.download_with_retry", _copy_download_stub(good, recorded)),
        pytest.raises(CheckpointValidationError) as excinfo,
    ):
        await source.process(CheckpointContext())

    assert "Checksum mismatch" in str(excinfo.value)


async def test_http_source_without_checksum_loads(tmp_path: Path) -> None:
    """With no expected checksum the download still loads (integrity unverified)."""
    good = _write_state_dict_ckpt(tmp_path / "good.ckpt")
    recorded: list[Path] = []

    source = HTTPSource(url="https://example.com/model.ckpt")
    with patch("anemoi.training.checkpoint.utils.download_with_retry", _copy_download_stub(good, recorded)):
        result = await source.process(CheckpointContext())

    assert result.checkpoint_data is not None


async def test_http_source_cleans_up_temp_file_on_load_error() -> None:
    """A corrupt download raises CheckpointLoadError and removes the temp file."""
    recorded: list[Path] = []

    source = HTTPSource(url="https://example.com/model.ckpt")
    with (
        patch("anemoi.training.checkpoint.utils.download_with_retry", _garbage_download_stub(recorded)),
        pytest.raises(CheckpointLoadError),
    ):
        await source.process(CheckpointContext())

    assert recorded, "download stub was not invoked"
    assert not recorded[0].exists()


# ---------------------------------------------------------------------------
# S3Source pure logic (URL parsing, resolution, supports) — no download/network
# ---------------------------------------------------------------------------


def test_s3_source_stores_url() -> None:
    source = S3Source(url="s3://my-bucket/checkpoints/model.ckpt")
    assert source.url == "s3://my-bucket/checkpoints/model.ckpt"


def test_s3_source_url_none_by_default() -> None:
    source = S3Source()
    assert source.url is None


def test_s3_resolve_url_constructor_takes_precedence() -> None:
    """A constructor URL wins over a URL present in context.config."""
    source = S3Source(url="s3://source1/model.ckpt")
    context = CheckpointContext(config={"url": "s3://source2/model.ckpt"})
    assert source._resolve_url(context) == "s3://source1/model.ckpt"


def test_s3_resolve_url_from_dict_config() -> None:
    source = S3Source()
    context = CheckpointContext(config={"url": "s3://bucket/key.ckpt"})
    assert source._resolve_url(context) == "s3://bucket/key.ckpt"


def test_s3_resolve_url_from_object_config() -> None:
    source = S3Source()
    context = CheckpointContext(config=OmegaConf.create({"url": "s3://bucket/key.ckpt"}))
    assert source._resolve_url(context) == "s3://bucket/key.ckpt"


def test_s3_resolve_url_missing_raises_config_error() -> None:
    source = S3Source()
    context = CheckpointContext(config=None)
    with pytest.raises(CheckpointConfigError) as excinfo:
        source._resolve_url(context)
    assert "URL" in str(excinfo.value)


def test_s3_parse_url_returns_bucket_and_key() -> None:
    bucket, key = S3Source._parse_s3_url("s3://my-bucket/path/to/model.ckpt")
    assert bucket == "my-bucket"
    assert key == "path/to/model.ckpt"


def test_s3_parse_url_strips_leading_slash_from_key() -> None:
    _, key = S3Source._parse_s3_url("s3://bucket/checkpoints/model.ckpt")
    assert key == "checkpoints/model.ckpt"


def test_s3_parse_url_rejects_non_s3_scheme() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        S3Source._parse_s3_url("s3a://bucket/key")
    assert "scheme" in str(excinfo.value)


def test_s3_parse_url_missing_bucket_raises() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        S3Source._parse_s3_url("s3:///model.ckpt")
    assert "bucket" in str(excinfo.value)


def test_s3_parse_url_missing_key_raises() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        S3Source._parse_s3_url("s3://bucket/")
    assert "key" in str(excinfo.value)


def test_s3_supports_s3_url() -> None:
    assert S3Source.supports("s3://bucket/model.ckpt") is True


def test_s3_supports_rejects_path_object() -> None:
    assert S3Source.supports(Path("/models/model.ckpt")) is False


def test_s3_supports_rejects_http_url() -> None:
    assert S3Source.supports("http://example.com/model.ckpt") is False


def test_s3_supports_rejects_local_path() -> None:
    assert S3Source.supports("/models/model.ckpt") is False


def test_s3_source_does_not_bind_anemoi_utils_at_module_level() -> None:
    """S3Source is importable/instantiable without the optional S3 dependency loaded.

    The download helper is imported lazily inside ``_download_from_s3``, so the
    module must not expose ``download_file`` at import time.
    """
    import anemoi.training.checkpoint.sources.s3 as s3_module

    assert not hasattr(s3_module, "download_file")
    # Construction must not require the optional dependency.
    assert S3Source(url="s3://bucket/key.ckpt").url == "s3://bucket/key.ckpt"

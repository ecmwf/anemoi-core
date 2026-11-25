# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for checkpoint I/O functions.

Covers save_metadata, load_metadata, has_metadata, extract_metadata_dict,
replace_metadata, remove_metadata, supporting arrays, legacy path detection,
and error handling.
"""

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.checkpoint import extract_metadata_dict
from anemoi.metadata.checkpoint import has_metadata
from anemoi.metadata.checkpoint import load_metadata
from anemoi.metadata.checkpoint import remove_metadata
from anemoi.metadata.checkpoint import replace_metadata
from anemoi.metadata.checkpoint import save_metadata
from anemoi.metadata.exceptions import CheckpointError
from anemoi.metadata.write_schema import COMPONENT_VERSIONS
from anemoi.metadata.write_schema import CURRENT_VERSION
from anemoi.metadata.write_schema import WriteValidationError
from anemoi.utils.checkpoints import DEFAULT_FOLDER
from anemoi.utils.checkpoints import DEFAULT_NAME
from anemoi.utils.checkpoints import DEPRECATED_NAME


@pytest.fixture
def f6(golden_loader: Callable[[str], dict[str, Any]]) -> dict:
    """The F6 golden metadata (what training main writes, version ``2.0``): a fresh dict per test.

    ``supporting_arrays_paths`` is dropped: it points at array files that these test checkpoints do not contain,
    so loading would fail.  The low-level writer sets it itself when arrays are given.
    """
    metadata = golden_loader("f6_main")
    del metadata["supporting_arrays_paths"]
    return metadata


@pytest.fixture
def f6_ckpt(empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> Path:
    """A checkpoint holding the F6 golden metadata, written unvalidated (it is historical content)."""
    ckpt = empty_ckpt()
    save_metadata(ckpt, f6, validate=False)
    return ckpt


def _without_array_paths(metadata: dict) -> dict:
    """Return *metadata* without ``supporting_arrays_paths``, which the low-level writer sets itself."""
    return {key: value for key, value in metadata.items() if key != "supporting_arrays_paths"}


# ---------------------------------------------------------------------------
# save_metadata / load_metadata
# ---------------------------------------------------------------------------


class TestSaveAndLoadMetadata:
    """save_metadata() creates a valid ZIP; load_metadata() reads it back."""

    def test_save_keeps_valid_zip(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """The checkpoint remains a valid ZIP archive after save."""
        ckpt = empty_ckpt()
        save_metadata(ckpt, f6, validate=False)
        assert zipfile.is_zipfile(ckpt)

    def test_save_contains_metadata_path(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """The ZIP contains an entry whose basename is the default metadata name.

        PyTorch checkpoints have a top-level directory prefix, so the full
        in-archive path is ``<dir>/anemoi-metadata/anemoi.json``.  We match
        by basename rather than exact path.
        """
        ckpt = empty_ckpt()
        save_metadata(ckpt, f6, validate=False)
        with zipfile.ZipFile(ckpt, "r") as zf:
            names = zf.namelist()
        assert any(Path(n).name == DEFAULT_NAME for n in names)

    def test_save_raises_if_file_missing(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """save_metadata() raises CheckpointError if the file doesn't exist."""
        with pytest.raises(CheckpointError, match="not found"):
            save_metadata(tmp_path / "ghost.ckpt", f6, validate=False)

    def test_load_returns_adapter_with_stored_version(self, f6_ckpt: Path) -> None:
        """load_metadata() returns a CheckpointAdapter that keeps the version written to the checkpoint."""
        meta = load_metadata(f6_ckpt)
        assert isinstance(meta, CheckpointAdapter)
        assert meta.version == "2.0"

    def test_save_raises_if_metadata_already_exists(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """save_metadata() raises CheckpointError if metadata already present."""
        with pytest.raises(CheckpointError, match="already contains metadata"):
            save_metadata(f6_ckpt, f6, validate=False)


# ---------------------------------------------------------------------------
# has_metadata
# ---------------------------------------------------------------------------


class TestHasMetadata:
    """has_metadata() correctly detects presence/absence of metadata."""

    def test_returns_true_for_checkpoint_with_metadata(self, f6_ckpt: Path) -> None:
        """has_metadata() returns True for a checkpoint with metadata."""
        assert has_metadata(f6_ckpt) is True

    def test_returns_false_for_empty_zip(self, tmp_path: Path) -> None:
        """has_metadata() returns False for a ZIP with no metadata entry."""
        ckpt = tmp_path / "empty.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("model_weights.pt", b"fake weights")
        assert has_metadata(ckpt) is False

    def test_returns_false_for_nonexistent_file(self, tmp_path: Path) -> None:
        """has_metadata() returns False when the file does not exist."""
        assert has_metadata(tmp_path / "nonexistent.ckpt") is False

    def test_raises_for_invalid_zip(self, tmp_path: Path) -> None:
        """has_metadata() raises CheckpointError for a corrupt file."""
        bad_file = tmp_path / "bad.ckpt"
        bad_file.write_bytes(b"this is not a zip file")
        with pytest.raises(CheckpointError):
            has_metadata(bad_file)


# ---------------------------------------------------------------------------
# extract_metadata_dict
# ---------------------------------------------------------------------------


class TestExtractMetadataDict:
    """extract_metadata_dict() returns the raw dict without validation."""

    def test_returns_the_stored_dict(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """extract_metadata_dict() returns the dict as written, as a plain dict with its version."""
        result = extract_metadata_dict(f6_ckpt)
        assert isinstance(result, dict)
        assert result["version"] == "2.0"
        assert _without_array_paths(result) == _without_array_paths(f6)

    def test_raises_for_missing_metadata(self, tmp_path: Path) -> None:
        """extract_metadata_dict() raises CheckpointError when no metadata."""
        ckpt = tmp_path / "no_meta.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("weights.pt", b"data")
        with pytest.raises(CheckpointError):
            extract_metadata_dict(ckpt)

    def test_raises_for_invalid_zip(self, tmp_path: Path) -> None:
        """extract_metadata_dict() raises CheckpointError for corrupt file."""
        bad = tmp_path / "bad.ckpt"
        bad.write_bytes(b"not a zip")
        with pytest.raises(CheckpointError):
            extract_metadata_dict(bad)


# ---------------------------------------------------------------------------
# replace_metadata
# ---------------------------------------------------------------------------


class TestReplaceMetadata:
    """replace_metadata() overwrites existing metadata."""

    def test_replace_updates_seed(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """replace_metadata() writes new metadata that can be read back."""
        f6["seed"] = 9999
        replace_metadata(f6_ckpt, CheckpointAdapter(f6), validate=False)
        loaded = load_metadata(f6_ckpt)
        assert loaded.sections.seeds["seed"] == 9999

    def test_replace_on_nonexistent_file_raises(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata() raises CheckpointError if file doesn't exist."""
        with pytest.raises(CheckpointError):
            replace_metadata(tmp_path / "ghost.ckpt", f6, validate=False)

    def test_replace_result_is_valid_zip(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """After replace_metadata(), the file is still a valid ZIP."""
        replace_metadata(f6_ckpt, f6, validate=False)
        assert zipfile.is_zipfile(f6_ckpt)


# ---------------------------------------------------------------------------
# remove_metadata
# ---------------------------------------------------------------------------


class TestRemoveMetadata:
    """remove_metadata() strips metadata entries from the ZIP."""

    def test_has_metadata_false_after_remove(self, f6_ckpt: Path) -> None:
        """has_metadata() returns False after remove_metadata(), and the file stays a valid ZIP."""
        remove_metadata(f6_ckpt)
        assert has_metadata(f6_ckpt) is False
        assert zipfile.is_zipfile(f6_ckpt)

    def test_other_entries_preserved_after_remove(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """Non-metadata entries are preserved after remove_metadata()."""
        ckpt = empty_ckpt()
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr(f"{ckpt.stem}/weights.pt", b"fake weights data")
        save_metadata(ckpt, f6, validate=False)

        remove_metadata(ckpt)

        with zipfile.ZipFile(ckpt, "r") as zf:
            names = zf.namelist()
        assert f"{ckpt.stem}/weights.pt" in names
        assert not any(Path(name).name == DEFAULT_NAME for name in names)

    def test_remove_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """remove_metadata() raises CheckpointError if file doesn't exist."""
        with pytest.raises(CheckpointError):
            remove_metadata(tmp_path / "ghost.ckpt")


# ---------------------------------------------------------------------------
# Supporting arrays
# ---------------------------------------------------------------------------


class TestSupportingArrays:
    """Numpy arrays round-trip through save/load with supporting_arrays."""

    def test_arrays_saved_and_loaded(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """Arrays stored with save_metadata() are returned by load_metadata()."""
        ckpt = empty_ckpt("model_with_arrays.ckpt")
        latitudes = np.linspace(-90, 90, 100)
        longitudes = np.linspace(-180, 180, 100)
        arrays = {"latitudes": latitudes, "longitudes": longitudes}

        save_metadata(ckpt, f6, supporting_arrays=arrays, validate=False)
        _, loaded_arrays = load_metadata(ckpt, supporting_arrays=True)

        assert "latitudes" in loaded_arrays
        assert "longitudes" in loaded_arrays

    def test_array_values_preserved(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """Array values are numerically identical after round-trip."""
        ckpt = empty_ckpt("model_arrays.ckpt")
        original = np.array([1.0, 2.5, 3.14, -0.5], dtype=np.float32)
        save_metadata(ckpt, f6, supporting_arrays={"weights": original}, validate=False)
        _, loaded_arrays = load_metadata(ckpt, supporting_arrays=True)
        np.testing.assert_array_equal(loaded_arrays["weights"], original)

    def test_empty_arrays_dict_no_extra_entries(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """Passing an empty arrays dict doesn't add spurious ZIP entries."""
        ckpt = empty_ckpt("model_empty_arrays.ckpt")
        save_metadata(ckpt, f6, supporting_arrays={}, validate=False)
        _, loaded_arrays = load_metadata(ckpt, supporting_arrays=True)
        assert loaded_arrays == {}


# ---------------------------------------------------------------------------
# Legacy path detection
# ---------------------------------------------------------------------------


class TestLegacyPath:
    """Checkpoints with ai-models.json are detected and loaded."""

    def _make_legacy_checkpoint(self, tmp_path: Path, data: dict) -> Path:
        """Create a ZIP with metadata at the legacy ai-models.json path."""
        ckpt = tmp_path / "legacy.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr(f"{DEFAULT_FOLDER}/{DEPRECATED_NAME}", json.dumps(data))
        return ckpt

    def test_has_metadata_detects_legacy(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """has_metadata() returns True for legacy ai-models.json path."""
        ckpt = self._make_legacy_checkpoint(tmp_path, f6)
        assert has_metadata(ckpt) is True

    def test_extract_dict_reads_legacy(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """extract_metadata_dict() reads from the legacy path."""
        ckpt = self._make_legacy_checkpoint(tmp_path, f6)
        result = extract_metadata_dict(ckpt)
        assert result["version"] == "2.0"

    def test_load_metadata_reads_legacy(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """load_metadata() loads from the legacy path."""
        ckpt = self._make_legacy_checkpoint(tmp_path, f6)
        meta = load_metadata(ckpt)
        assert isinstance(meta, CheckpointAdapter)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    """Checkpoint functions raise CheckpointError for invalid inputs."""

    def test_load_invalid_zip_raises(self, tmp_path: Path) -> None:
        """load_metadata() raises CheckpointError for a corrupt file."""
        bad = tmp_path / "corrupt.ckpt"
        bad.write_bytes(b"not a zip archive at all")
        with pytest.raises(CheckpointError):
            load_metadata(bad)

    def test_load_missing_metadata_raises(self, tmp_path: Path) -> None:
        """load_metadata() raises CheckpointError when no metadata entry."""
        ckpt = tmp_path / "no_meta.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("weights.pt", b"data")
        with pytest.raises(CheckpointError):
            load_metadata(ckpt)

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """load_metadata() raises CheckpointError for malformed JSON."""
        ckpt = tmp_path / "bad_json.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr(f"{DEFAULT_FOLDER}/{DEFAULT_NAME}", "{ this is not valid json }")
        with pytest.raises(CheckpointError):
            load_metadata(ckpt)


# ---------------------------------------------------------------------------
# Content written unchanged
# ---------------------------------------------------------------------------


class TestWriteContent:
    """With validate=False the writer stores the raw content unchanged; it does not stamp or reshape it."""

    def test_save_dict_without_version_writes_no_version(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """A dict without a version is written without one."""
        del f6["version"]
        ckpt = tmp_path / "no_version.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, f6, validate=False)

        result = extract_metadata_dict(ckpt)
        assert "version" not in result
        # The low-level writer sets supporting_arrays_paths; everything else is unchanged.
        assert _without_array_paths(result) == _without_array_paths(f6)

    def test_save_preserves_explicit_version(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """save_metadata does not overwrite an explicitly set version."""
        f6["version"] = "3.0"
        ckpt = tmp_path / "explicit_version.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, f6, validate=False)

        assert extract_metadata_dict(ckpt)["version"] == "3.0"

    def test_save_preserves_unknown_keys(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """Unknown keys survive the write."""
        f6["future_key"] = {"a": [1, 2]}
        ckpt = tmp_path / "unknown_keys.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, f6, validate=False)

        assert extract_metadata_dict(ckpt)["future_key"] == {"a": [1, 2]}

    def test_save_accepts_checkpoint_adapter(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """save_metadata accepts a CheckpointAdapter instance directly."""
        ckpt = tmp_path / "adapter.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, CheckpointAdapter(f6), validate=False)

        result = extract_metadata_dict(ckpt)
        assert result["version"] == "2.0"

    def test_save_rejects_non_dict_non_adapter(self, tmp_path: Path) -> None:
        """save_metadata raises TypeError for unsupported types."""
        ckpt = tmp_path / "bad.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        with pytest.raises(TypeError, match="CheckpointAdapter"):
            save_metadata(ckpt, "not valid metadata")

    def test_replace_does_not_change_version(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """replace_metadata writes the version it is given."""
        f6["version"] = "3.0"
        replace_metadata(f6_ckpt, f6, validate=False)
        assert extract_metadata_dict(f6_ckpt)["version"] == "3.0"

    def test_replace_without_version_raises(self, f6_ckpt: Path, f6: dict[str, Any]) -> None:
        """The checkpoint writer needs a version; the package does not invent one."""
        del f6["version"]
        with pytest.raises(CheckpointError, match="version"):
            replace_metadata(f6_ckpt, f6, validate=False)


# ---------------------------------------------------------------------------
# Strict write (default)
# ---------------------------------------------------------------------------


@pytest.fixture
def f9(golden_loader: Callable[[str], dict[str, Any]]) -> dict:
    """An F9 dict: the F8 golden file stamped as written by this package."""
    metadata = golden_loader("f8_richer_batch")
    metadata["version"] = CURRENT_VERSION
    metadata["component_versions"] = dict(COMPONENT_VERSIONS)
    return metadata


class TestStrictWrite:
    """save_metadata and replace_metadata validate by default."""

    def test_save_accepts_adapter(self, empty_ckpt: Callable[..., Path], f9: dict[str, Any]) -> None:
        """A CheckpointAdapter over valid F9 content is validated and written."""
        ckpt = empty_ckpt()
        save_metadata(ckpt, CheckpointAdapter(f9))
        assert extract_metadata_dict(ckpt)["version"] == CURRENT_VERSION

    def test_save_invalid_raises_and_writes_nothing(self, empty_ckpt: Callable[..., Path], f9: dict[str, Any]) -> None:
        """An invalid dict raises WriteValidationError and leaves the checkpoint without metadata."""
        del f9["metadata_inference"]["dataset_names"]
        ckpt = empty_ckpt()
        with pytest.raises(WriteValidationError, match="dataset_names"):
            save_metadata(ckpt, f9)
        assert has_metadata(ckpt) is False

    def test_save_historical_dict_rejected(self, empty_ckpt: Callable[..., Path], f6: dict[str, Any]) -> None:
        """A historical 2.0 dict is never relabelled as current."""
        ckpt = empty_ckpt()
        with pytest.raises(WriteValidationError, match="version"):
            save_metadata(ckpt, f6)
        assert has_metadata(ckpt) is False

    def test_save_validate_false_writes_unchanged(self, empty_ckpt: Callable[..., Path], f9: dict[str, Any]) -> None:
        """validate=False writes the dict as it is, even when invalid."""
        del f9["component_versions"]
        ckpt = empty_ckpt()
        save_metadata(ckpt, f9, validate=False)
        stored = extract_metadata_dict(ckpt)
        assert "component_versions" not in stored
        assert stored["version"] == CURRENT_VERSION

    def test_replace_valid_f9(self, f6_ckpt: Path, f9: dict[str, Any]) -> None:
        """replace_metadata writes a valid F9 dict with the current version."""
        replace_metadata(f6_ckpt, f9)
        stored = extract_metadata_dict(f6_ckpt)
        assert stored["version"] == CURRENT_VERSION
        assert stored["component_versions"] == COMPONENT_VERSIONS

    def test_replace_without_version_is_stamped(self, f6_ckpt: Path, f9: dict[str, Any]) -> None:
        """A missing version is set by the strict writer (the low-level writer would refuse)."""
        del f9["version"]
        replace_metadata(f6_ckpt, f9)
        assert extract_metadata_dict(f6_ckpt)["version"] == CURRENT_VERSION

    def test_replace_invalid_leaves_checkpoint_untouched(self, f6_ckpt: Path, f9: dict[str, Any]) -> None:
        """A rejected replace keeps the stored metadata."""
        f9["metadata_inference"]["data"]["data_type"] = "image"
        with pytest.raises(WriteValidationError, match="data_type"):
            replace_metadata(f6_ckpt, f9)
        assert extract_metadata_dict(f6_ckpt)["version"] == "2.0"


class TestLegacyFlatPathReplace:
    """Bug fix: replace_metadata on legacy flat checkpoints (archive/ai-models.json)."""

    def _make_legacy_flat_checkpoint(self, tmp_path: Path, metadata: dict) -> Path:
        """Create a legacy checkpoint with flat metadata path and test data.

        This mimics the deprecated structure where metadata is at
        ``archive/ai-models.json`` (one level deep, not two).
        """
        ckpt = tmp_path / "legacy_flat.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            # Metadata directly under top-level dir (deprecated flat structure).
            zf.writestr("archive/ai-models.json", json.dumps(metadata))
            # Some weights file.
            zf.writestr("archive/data/weights.bin", b"fake weights data")
            # An unrelated .numpy file that should NOT be deleted.
            zf.writestr("archive/data/something.numpy", b"unrelated array")
        return ckpt

    def test_replace_on_legacy_flat_no_leading_slash(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata on legacy flat checkpoint produces no leading-slash members."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, f6)
        replace_metadata(ckpt, f6, validate=False)

        with zipfile.ZipFile(ckpt, "r") as zf:
            for name in zf.namelist():
                assert not name.startswith("/"), f"Found leading slash in: {name}"

    def test_replace_on_legacy_flat_moves_to_canonical_path(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata on legacy flat checkpoint writes to the canonical path."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, f6)
        replace_metadata(ckpt, f6, validate=False)

        with zipfile.ZipFile(ckpt, "r") as zf:
            # New metadata should be at archive/anemoi-metadata/anemoi.json.
            assert any("anemoi-metadata/anemoi.json" in name for name in zf.namelist())
            # Old legacy path should be gone.
            assert "archive/ai-models.json" not in zf.namelist()

    def test_replace_on_legacy_flat_preserves_unrelated_numpy(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata on legacy flat checkpoint preserves unrelated .numpy files."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, f6)
        replace_metadata(ckpt, f6, validate=False)

        with zipfile.ZipFile(ckpt, "r") as zf:
            # The unrelated .numpy file should still exist.
            assert "archive/data/something.numpy" in zf.namelist()

    def test_replace_on_legacy_flat_preserves_single_top_dir(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata on legacy flat checkpoint maintains single top-level dir."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, f6)
        replace_metadata(ckpt, f6, validate=False)

        # Should still load successfully (validates single top-level directory).
        loaded = load_metadata(ckpt)
        assert loaded.sections.seeds["seed"] == f6["seed"]

    def test_replace_on_legacy_flat_preserves_weights(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata on legacy flat checkpoint preserves other checkpoint data."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, f6)
        replace_metadata(ckpt, f6, validate=False)

        with zipfile.ZipFile(ckpt, "r") as zf:
            assert "archive/data/weights.bin" in zf.namelist()


class TestReplaceMetadataSupportingArrays:
    """Bug fix: replace_metadata removes only referenced arrays, not all .numpy files."""

    def _make_checkpoint_with_arrays(self, tmp_path: Path, metadata: dict, arrays: dict) -> Path:
        """Create a checkpoint with metadata and supporting arrays."""
        ckpt = tmp_path / "with_arrays.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            # Write some dummy checkpoint data.
            zf.writestr("model/data.pkl", b"")

        # Use save_metadata to write metadata with arrays (it handles paths).
        save_metadata(ckpt, metadata, supporting_arrays=arrays, validate=False)
        return ckpt

    def test_replace_removes_only_referenced_arrays(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata removes only arrays referenced in old metadata."""
        # Create a checkpoint with supporting arrays.
        arrays = {"latitudes": np.linspace(-90, 90, 10)}
        ckpt = self._make_checkpoint_with_arrays(tmp_path, f6, arrays)

        # Add an unrelated .numpy file in a different location.
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr("model/unrelated/data.numpy", b"unrelated data")

        # Replace with new metadata (no arrays).
        f6["seed"] = 9999
        replace_metadata(ckpt, f6, validate=False)

        # The old array should be gone, unrelated .numpy should remain.
        with zipfile.ZipFile(ckpt, "r") as zf:
            names = zf.namelist()
            # Old array path should be gone.
            assert not any("latitudes.numpy" in n for n in names)
            # Unrelated .numpy should still exist.
            assert "model/unrelated/data.numpy" in names

    def test_replace_preserves_unrelated_numpy_files(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """replace_metadata preserves .numpy files not referenced in metadata."""
        # Create checkpoint without arrays.
        ckpt = tmp_path / "no_arrays.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("model/data.pkl", b"")
        save_metadata(ckpt, f6, validate=False)

        # Add an unrelated .numpy file.
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr("model/something.numpy", b"unrelated")

        # Replace metadata.
        f6["seed"] = 8888
        replace_metadata(ckpt, f6, validate=False)

        # Unrelated .numpy should survive.
        with zipfile.ZipFile(ckpt, "r") as zf:
            assert "model/something.numpy" in zf.namelist()


class TestPrefixCollisionGuard:
    """Bug fix: directory prefix matching uses exact boundaries."""

    def test_prefix_collision_avoided(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """Replacing metadata doesn't delete files in similarly-named directories."""
        # Create a checkpoint with metadata.
        ckpt = tmp_path / "prefix.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pkl", b"")
        save_metadata(ckpt, f6, validate=False)

        # Add a .numpy file under a directory with a similar prefix.
        with zipfile.ZipFile(ckpt, "a") as zf:
            # If old metadata is at "archive/anemoi-metadata/anemoi.json",
            # this should NOT match "archive/anemoi-metadata/..." prefix.
            zf.writestr("archive/anemoi-metadata-old/foo.numpy", b"unrelated")

        # Replace metadata.
        f6["seed"] = 7777

        replace_metadata(ckpt, f6, validate=False)

        # The similarly-named directory's file should survive.
        with zipfile.ZipFile(ckpt, "r") as zf:
            assert "archive/anemoi-metadata-old/foo.numpy" in zf.namelist()


class TestEditCommandFailure:
    """Bug fix: edit command preserves user edits when editor exits non-zero."""

    def test_editor_failure_preserves_edited_file(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """When editor exits non-zero, edited file is preserved to a persistent location."""
        import copy

        from anemoi.metadata.commands.edit import _preserve_edited_file
        from anemoi.metadata.commands.edit import _serialise_to_string

        # Create a temporary file with edited content.
        tmp_file = tmp_path / "edited.json"
        edited_data = copy.deepcopy(f6)
        edited_data["metadata_inference"]["seed"] = 12345

        with tmp_file.open("w") as f:
            json.dump(edited_data, f, indent=4, sort_keys=True)

        # Original serialised content (different from edited).
        original = _serialise_to_string(f6, use_yaml=False)

        # Preserve the edited file.
        preserved = _preserve_edited_file(str(tmp_file), "json", original)

        try:
            assert preserved is not None
            assert Path(preserved).exists()
            assert "anemoi-metadata-edit-" in preserved

            # Verify content was preserved.
            with Path(preserved).open() as f:
                preserved_data = json.load(f)
            assert preserved_data["metadata_inference"]["seed"] == 12345
        finally:
            # Clean up.
            if preserved and Path(preserved).exists():
                Path(preserved).unlink()

    def test_editor_failure_no_preservation_if_unchanged(self, tmp_path: Path, f6: dict[str, Any]) -> None:
        """Edited file is not preserved if content matches original."""
        from anemoi.metadata.commands.edit import _preserve_edited_file
        from anemoi.metadata.commands.edit import _serialise_to_string

        tmp_file = tmp_path / "unchanged.json"
        original = _serialise_to_string(f6, use_yaml=False)

        with tmp_file.open("w") as f:
            f.write(original)

        # Try to preserve (should return None because content is unchanged).
        preserved = _preserve_edited_file(str(tmp_file), "json", original)
        assert preserved is None

    def test_editor_failure_no_preservation_if_file_missing(self, tmp_path: Path) -> None:
        """No preservation if temporary file doesn't exist."""
        from anemoi.metadata.commands.edit import _preserve_edited_file

        preserved = _preserve_edited_file(str(tmp_path / "nonexistent.json"), "json", "original content")
        assert preserved is None

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
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from anemoi.metadata.checkpoint import DEFAULT_NAME
from anemoi.metadata.checkpoint import LEGACY_METADATA_PATH
from anemoi.metadata.checkpoint import METADATA_PATH
from anemoi.metadata.checkpoint import extract_metadata_dict
from anemoi.metadata.checkpoint import has_metadata
from anemoi.metadata.checkpoint import load_metadata
from anemoi.metadata.checkpoint import remove_metadata
from anemoi.metadata.checkpoint import replace_metadata
from anemoi.metadata.checkpoint import save_metadata
from anemoi.metadata.exceptions import CheckpointError
from anemoi.metadata.registry import MetadataRegistry
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# save_metadata / load_metadata
# ---------------------------------------------------------------------------


class TestSaveAndLoadMetadata:
    """save_metadata() creates a valid ZIP; load_metadata() reads it back."""

    def test_save_writes_to_existing_checkpoint(self, empty_ckpt, sample_metadata_v1):
        """save_metadata() leaves the checkpoint in place after writing."""
        ckpt = empty_ckpt()
        save_metadata(ckpt, sample_metadata_v1)
        assert ckpt.exists()

    def test_save_keeps_valid_zip(self, empty_ckpt, sample_metadata_v1):
        """The checkpoint remains a valid ZIP archive after save."""
        ckpt = empty_ckpt()
        save_metadata(ckpt, sample_metadata_v1)
        assert zipfile.is_zipfile(ckpt)

    def test_save_contains_metadata_path(self, empty_ckpt, sample_metadata_v1):
        """The ZIP contains an entry whose basename is the default metadata name.

        PyTorch checkpoints have a top-level directory prefix, so the full
        in-archive path is ``<dir>/anemoi-metadata/anemoi.json``.  We match
        by basename rather than exact path.
        """
        ckpt = empty_ckpt()
        save_metadata(ckpt, sample_metadata_v1)
        with zipfile.ZipFile(ckpt, "r") as zf:
            names = zf.namelist()
        assert any(os.path.basename(n) == DEFAULT_NAME for n in names)

    def test_save_raises_if_file_missing(self, tmp_path, sample_metadata_v1):
        """save_metadata() raises CheckpointError if the file doesn't exist."""
        with pytest.raises(CheckpointError, match="not found"):
            save_metadata(tmp_path / "ghost.ckpt", sample_metadata_v1)

    def test_load_returns_metadata_v1(self, tmp_checkpoint):
        """load_metadata() returns MetadataV1 (only version registered)."""
        from anemoi.metadata.versions.v1 import MetadataV1

        meta = load_metadata(tmp_checkpoint)
        assert isinstance(meta, MetadataV1)

    def test_load_migrated_schema_version(self, tmp_checkpoint):
        """Loaded metadata has the latest schema_version after auto-migration."""

        meta = load_metadata(tmp_checkpoint)
        assert meta.schema_version == MetadataRegistry.latest_version()

    def test_load_preserves_seed(self, tmp_checkpoint):
        """Loaded metadata preserves the seed value."""
        meta = load_metadata(tmp_checkpoint)
        assert meta.metadata_inference.seed == 42

    def test_load_preserves_dataset_names(self, tmp_checkpoint):
        """Loaded metadata preserves dataset_names."""
        meta = load_metadata(tmp_checkpoint)
        assert "era5_1deg" in meta.metadata_inference.dataset_names

    def test_load_typed_inference_metadata(self, tmp_checkpoint):
        """Loaded metadata has a fully typed InferenceMetadata block."""
        from anemoi.metadata.versions.v1 import InferenceMetadata

        meta = load_metadata(tmp_checkpoint)
        assert isinstance(meta.metadata_inference, InferenceMetadata)

    def test_load_migrate_false(self, tmp_checkpoint):
        """load_metadata() with migrate=False still returns MetadataV1."""
        meta = load_metadata(tmp_checkpoint, migrate=False)
        assert isinstance(meta, MetadataV1)

    def test_save_raises_if_metadata_already_exists(self, tmp_checkpoint, sample_metadata_v1):
        """save_metadata() raises CheckpointError if metadata already present."""
        with pytest.raises(CheckpointError, match="already contains metadata"):
            save_metadata(tmp_checkpoint, sample_metadata_v1)


# ---------------------------------------------------------------------------
# has_metadata
# ---------------------------------------------------------------------------


class TestHasMetadata:
    """has_metadata() correctly detects presence/absence of metadata."""

    def test_returns_true_for_checkpoint_with_metadata(self, tmp_checkpoint):
        """has_metadata() returns True for a checkpoint with metadata."""
        assert has_metadata(tmp_checkpoint) is True

    def test_returns_false_for_empty_zip(self, tmp_path):
        """has_metadata() returns False for a ZIP with no metadata entry."""
        ckpt = tmp_path / "empty.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("model_weights.pt", b"fake weights")
        assert has_metadata(ckpt) is False

    def test_returns_false_for_nonexistent_file(self, tmp_path):
        """has_metadata() returns False when the file does not exist."""
        assert has_metadata(tmp_path / "nonexistent.ckpt") is False

    def test_raises_for_invalid_zip(self, tmp_path):
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

    def test_returns_dict(self, tmp_checkpoint):
        """extract_metadata_dict() returns a dict."""
        result = extract_metadata_dict(tmp_checkpoint)
        assert isinstance(result, dict)

    def test_contains_schema_version(self, tmp_checkpoint):
        """Extracted dict contains schema_version."""
        result = extract_metadata_dict(tmp_checkpoint)
        assert result["schema_version"] == "1.0"

    def test_contains_metadata_inference(self, tmp_checkpoint):
        """Extracted dict contains metadata_inference key."""
        result = extract_metadata_dict(tmp_checkpoint)
        assert "metadata_inference" in result

    def test_raises_for_missing_metadata(self, tmp_path):
        """extract_metadata_dict() raises CheckpointError when no metadata."""
        ckpt = tmp_path / "no_meta.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("weights.pt", b"data")
        with pytest.raises(CheckpointError):
            extract_metadata_dict(ckpt)

    def test_raises_for_invalid_zip(self, tmp_path):
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

    def test_replace_updates_seed(self, tmp_checkpoint, sample_v1_dict):
        """replace_metadata() writes new metadata that can be read back."""
        # Build a modified V1 with a different seed
        modified = dict(sample_v1_dict)
        modified["metadata_inference"] = dict(sample_v1_dict["metadata_inference"])
        modified["metadata_inference"]["seed"] = 9999
        new_meta = MetadataV1.model_validate(modified)

        replace_metadata(tmp_checkpoint, new_meta)
        loaded = load_metadata(tmp_checkpoint)
        assert loaded.metadata_inference.seed == 9999

    def test_replace_on_nonexistent_file_raises(self, tmp_path, sample_metadata_v1):
        """replace_metadata() raises CheckpointError if file doesn't exist."""
        with pytest.raises(CheckpointError):
            replace_metadata(tmp_path / "ghost.ckpt", sample_metadata_v1)

    def test_replace_result_is_valid_zip(self, tmp_checkpoint, sample_metadata_v1):
        """After replace_metadata(), the file is still a valid ZIP."""
        replace_metadata(tmp_checkpoint, sample_metadata_v1)
        assert zipfile.is_zipfile(tmp_checkpoint)


# ---------------------------------------------------------------------------
# remove_metadata
# ---------------------------------------------------------------------------


class TestRemoveMetadata:
    """remove_metadata() strips metadata entries from the ZIP."""

    def test_has_metadata_false_after_remove(self, tmp_checkpoint):
        """has_metadata() returns False after remove_metadata()."""
        remove_metadata(tmp_checkpoint)
        assert has_metadata(tmp_checkpoint) is False

    def test_file_still_valid_zip_after_remove(self, tmp_checkpoint):
        """The file remains a valid ZIP after remove_metadata()."""
        remove_metadata(tmp_checkpoint)
        assert zipfile.is_zipfile(tmp_checkpoint)

    def test_other_entries_preserved_after_remove(self, empty_ckpt, sample_metadata_v1):
        """Non-metadata entries are preserved after remove_metadata()."""
        ckpt = empty_ckpt()
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr(f"{ckpt.stem}/weights.pt", b"fake weights data")
        save_metadata(ckpt, sample_metadata_v1)

        remove_metadata(ckpt)

        with zipfile.ZipFile(ckpt, "r") as zf:
            assert f"{ckpt.stem}/weights.pt" in zf.namelist()
            assert METADATA_PATH not in zf.namelist()

    def test_remove_nonexistent_file_raises(self, tmp_path):
        """remove_metadata() raises CheckpointError if file doesn't exist."""
        with pytest.raises(CheckpointError):
            remove_metadata(tmp_path / "ghost.ckpt")


# ---------------------------------------------------------------------------
# Supporting arrays
# ---------------------------------------------------------------------------


class TestSupportingArrays:
    """Numpy arrays round-trip through save/load with supporting_arrays."""

    def test_arrays_saved_and_loaded(self, empty_ckpt, sample_metadata_v1):
        """Arrays stored with save_metadata() are returned by load_metadata()."""
        ckpt = empty_ckpt("model_with_arrays.ckpt")
        latitudes = np.linspace(-90, 90, 100)
        longitudes = np.linspace(-180, 180, 100)
        arrays = {"latitudes": latitudes, "longitudes": longitudes}

        save_metadata(ckpt, sample_metadata_v1, supporting_arrays=arrays)
        _, loaded_arrays = load_metadata(ckpt, supporting_arrays=True)

        assert "latitudes" in loaded_arrays
        assert "longitudes" in loaded_arrays

    def test_array_values_preserved(self, empty_ckpt, sample_metadata_v1):
        """Array values are numerically identical after round-trip."""
        ckpt = empty_ckpt("model_arrays.ckpt")
        original = np.array([1.0, 2.5, 3.14, -0.5], dtype=np.float32)
        save_metadata(ckpt, sample_metadata_v1, supporting_arrays={"weights": original})
        _, loaded_arrays = load_metadata(ckpt, supporting_arrays=True)
        np.testing.assert_array_equal(loaded_arrays["weights"], original)

    def test_no_arrays_returns_metadata_only(self, tmp_checkpoint):
        """load_metadata() without supporting_arrays returns metadata only."""
        from anemoi.metadata.base import MetadataContract

        result = load_metadata(tmp_checkpoint)
        # Should be a MetadataContract subclass, not a tuple
        assert isinstance(result, MetadataContract)

    def test_empty_arrays_dict_no_extra_entries(self, empty_ckpt, sample_metadata_v1):
        """Passing an empty arrays dict doesn't add spurious ZIP entries."""
        ckpt = empty_ckpt("model_empty_arrays.ckpt")
        save_metadata(ckpt, sample_metadata_v1, supporting_arrays={})
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
            zf.writestr(LEGACY_METADATA_PATH, json.dumps(data))
        return ckpt

    def test_has_metadata_detects_legacy(self, tmp_path, sample_v1_dict):
        """has_metadata() returns True for legacy ai-models.json path."""
        ckpt = self._make_legacy_checkpoint(tmp_path, sample_v1_dict)
        assert has_metadata(ckpt) is True

    def test_extract_dict_reads_legacy(self, tmp_path, sample_v1_dict):
        """extract_metadata_dict() reads from the legacy path."""
        ckpt = self._make_legacy_checkpoint(tmp_path, sample_v1_dict)
        result = extract_metadata_dict(ckpt)
        assert result["schema_version"] == "1.0"

    def test_load_metadata_reads_legacy(self, tmp_path, sample_v1_dict):
        """load_metadata() loads and validates from the legacy path."""
        from anemoi.metadata.base import MetadataContract

        ckpt = self._make_legacy_checkpoint(tmp_path, sample_v1_dict)
        meta = load_metadata(ckpt)
        assert isinstance(meta, MetadataContract)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    """Checkpoint functions raise CheckpointError for invalid inputs."""

    def test_load_invalid_zip_raises(self, tmp_path):
        """load_metadata() raises CheckpointError for a corrupt file."""
        bad = tmp_path / "corrupt.ckpt"
        bad.write_bytes(b"not a zip archive at all")
        with pytest.raises(CheckpointError):
            load_metadata(bad)

    def test_load_missing_metadata_raises(self, tmp_path):
        """load_metadata() raises CheckpointError when no metadata entry."""
        ckpt = tmp_path / "no_meta.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("weights.pt", b"data")
        with pytest.raises(CheckpointError):
            load_metadata(ckpt)

    def test_load_invalid_json_raises(self, tmp_path):
        """load_metadata() raises CheckpointError for malformed JSON."""
        ckpt = tmp_path / "bad_json.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr(METADATA_PATH, "{ this is not valid json }")
        with pytest.raises(CheckpointError):
            load_metadata(ckpt)

    def test_has_metadata_nonexistent_returns_false(self, tmp_path):
        """has_metadata() returns False (not an error) for missing files."""
        result = has_metadata(tmp_path / "does_not_exist.ckpt")
        assert result is False


# ---------------------------------------------------------------------------
# schema_version enforcement on write
# ---------------------------------------------------------------------------


class TestSchemaVersionOnWrite:
    """save_metadata validates dicts through the registry on write."""

    def test_save_validates_dict_and_sets_schema_version(self, tmp_path, sample_v1_dict):
        """save_metadata validates a raw dict, ensuring schema_version is written.

        A dict without ``schema_version`` but with ``metadata_inference`` is
        treated as a transitional V1 checkpoint and saved with
        ``schema_version="1.0"``.
        """
        del sample_v1_dict["schema_version"]
        ckpt = tmp_path / "no_sv.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        with pytest.warns(UserWarning, match="schema_version"):
            save_metadata(ckpt, sample_v1_dict)

        result = extract_metadata_dict(ckpt)
        assert "schema_version" in result
        assert result["schema_version"] == "1.0"

    def test_save_preserves_explicit_schema_version(self, tmp_path, sample_v1_dict):
        """save_metadata does not overwrite an explicitly set schema_version."""
        sample_v1_dict["schema_version"] = "1.0"
        ckpt = tmp_path / "explicit_sv.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, sample_v1_dict)

        result = extract_metadata_dict(ckpt)
        assert result["schema_version"] == "1.0"

    def test_save_accepts_metadata_contract(self, tmp_path, sample_metadata_v1):
        """save_metadata accepts a MetadataContract instance directly."""
        ckpt = tmp_path / "contract.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        save_metadata(ckpt, sample_metadata_v1)

        result = extract_metadata_dict(ckpt)
        assert result["schema_version"] == "1.0"

    def test_save_rejects_non_dict_non_contract(self, tmp_path):
        """save_metadata raises TypeError for unsupported types."""
        ckpt = tmp_path / "bad.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pt", b"weights")

        with pytest.raises(TypeError, match="MetadataContract"):
            save_metadata(ckpt, "not valid metadata")


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

    def test_replace_on_legacy_flat_no_leading_slash(self, tmp_path, sample_metadata_v1):
        """replace_metadata on legacy flat checkpoint produces no leading-slash members."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, sample_metadata_v1.to_dict())
        replace_metadata(ckpt, sample_metadata_v1)

        with zipfile.ZipFile(ckpt, "r") as zf:
            for name in zf.namelist():
                assert not name.startswith("/"), f"Found leading slash in: {name}"

    def test_replace_on_legacy_flat_migrates_to_canonical_path(self, tmp_path, sample_metadata_v1):
        """replace_metadata on legacy flat checkpoint migrates to canonical path."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, sample_metadata_v1.to_dict())
        replace_metadata(ckpt, sample_metadata_v1)

        with zipfile.ZipFile(ckpt, "r") as zf:
            # New metadata should be at archive/anemoi-metadata/anemoi.json.
            assert any("anemoi-metadata/anemoi.json" in name for name in zf.namelist())
            # Old legacy path should be gone.
            assert "archive/ai-models.json" not in zf.namelist()

    def test_replace_on_legacy_flat_preserves_unrelated_numpy(self, tmp_path, sample_metadata_v1):
        """replace_metadata on legacy flat checkpoint preserves unrelated .numpy files."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, sample_metadata_v1.to_dict())
        replace_metadata(ckpt, sample_metadata_v1)

        with zipfile.ZipFile(ckpt, "r") as zf:
            # The unrelated .numpy file should still exist.
            assert "archive/data/something.numpy" in zf.namelist()

    def test_replace_on_legacy_flat_preserves_single_top_dir(self, tmp_path, sample_metadata_v1):
        """replace_metadata on legacy flat checkpoint maintains single top-level dir."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, sample_metadata_v1.to_dict())
        replace_metadata(ckpt, sample_metadata_v1)

        # Should still load successfully (validates single top-level directory).
        loaded = load_metadata(ckpt)
        assert loaded.metadata_inference.seed == 42

    def test_replace_on_legacy_flat_preserves_weights(self, tmp_path, sample_metadata_v1):
        """replace_metadata on legacy flat checkpoint preserves other checkpoint data."""
        ckpt = self._make_legacy_flat_checkpoint(tmp_path, sample_metadata_v1.to_dict())
        replace_metadata(ckpt, sample_metadata_v1)

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
        save_metadata(ckpt, metadata, supporting_arrays=arrays)
        return ckpt

    def test_replace_removes_only_referenced_arrays(self, tmp_path, sample_metadata_v1):
        """replace_metadata removes only arrays referenced in old metadata."""
        # Create a checkpoint with supporting arrays.
        arrays = {"latitudes": np.linspace(-90, 90, 10)}
        ckpt = self._make_checkpoint_with_arrays(tmp_path, sample_metadata_v1, arrays)

        # Add an unrelated .numpy file in a different location.
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr("model/unrelated/data.numpy", b"unrelated data")

        # Replace with new metadata (no arrays).
        modified = sample_metadata_v1.to_dict()
        modified["metadata_inference"]["seed"] = 9999
        replace_metadata(ckpt, modified)

        # The old array should be gone, unrelated .numpy should remain.
        with zipfile.ZipFile(ckpt, "r") as zf:
            names = zf.namelist()
            # Old array path should be gone.
            assert not any("latitudes.numpy" in n for n in names)
            # Unrelated .numpy should still exist.
            assert "model/unrelated/data.numpy" in names

    def test_replace_preserves_unrelated_numpy_files(self, tmp_path, sample_metadata_v1):
        """replace_metadata preserves .numpy files not referenced in metadata."""
        # Create checkpoint without arrays.
        ckpt = tmp_path / "no_arrays.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("model/data.pkl", b"")
        save_metadata(ckpt, sample_metadata_v1)

        # Add an unrelated .numpy file.
        with zipfile.ZipFile(ckpt, "a") as zf:
            zf.writestr("model/something.numpy", b"unrelated")

        # Replace metadata.
        modified = sample_metadata_v1.to_dict()
        modified["metadata_inference"]["seed"] = 8888
        replace_metadata(ckpt, modified)

        # Unrelated .numpy should survive.
        with zipfile.ZipFile(ckpt, "r") as zf:
            assert "model/something.numpy" in zf.namelist()


class TestPrefixCollisionGuard:
    """Bug fix: directory prefix matching uses exact boundaries."""

    def test_prefix_collision_avoided(self, tmp_path, sample_metadata_v1):
        """Replacing metadata doesn't delete files in similarly-named directories."""
        # Create a checkpoint with metadata.
        ckpt = tmp_path / "prefix.ckpt"
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pkl", b"")
        save_metadata(ckpt, sample_metadata_v1)

        # Add a .numpy file under a directory with a similar prefix.
        with zipfile.ZipFile(ckpt, "a") as zf:
            # If old metadata is at "archive/anemoi-metadata/anemoi.json",
            # this should NOT match "archive/anemoi-metadata/..." prefix.
            zf.writestr("archive/anemoi-metadata-old/foo.numpy", b"unrelated")

        # Replace metadata.
        modified = sample_metadata_v1.to_dict()
        modified["metadata_inference"]["seed"] = 7777
        replace_metadata(ckpt, modified)

        # The similarly-named directory's file should survive.
        with zipfile.ZipFile(ckpt, "r") as zf:
            assert "archive/anemoi-metadata-old/foo.numpy" in zf.namelist()

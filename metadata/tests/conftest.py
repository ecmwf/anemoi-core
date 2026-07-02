# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Shared fixtures for the anemoi.metadata test suite.

Provides realistic metadata structures that mirror what anemoi-training
writes to checkpoint files.
"""

import json
import zipfile

import pytest

from anemoi.metadata.checkpoint import METADATA_PATH
from anemoi.metadata.versions.v1 import MetadataV1

# ---------------------------------------------------------------------------
# Realistic per-dataset inference data (flat form, as training writes it)
# ---------------------------------------------------------------------------


def _make_dataset_block(name: str, var_offset: int = 0) -> dict:
    """Build a realistic per-dataset sub-dict for the flat checkpoint format.

    Parameters
    ----------
    name : str
        Dataset name (used as the key in the flat dict).
    var_offset : int, optional
        Index offset so that two datasets can have non-overlapping indices.

    Returns
    -------
    dict
        Per-dataset block matching what anemoi-training writes.
    """
    return {
        "data_indices": {
            "input": {
                "2t": var_offset + 0,
                "msl": var_offset + 1,
                "10u": var_offset + 2,
                "10v": var_offset + 3,
                "lsm": var_offset + 4,
            },
            "output": {
                "2t": var_offset + 0,
                "msl": var_offset + 1,
                "10u": var_offset + 2,
                "10v": var_offset + 3,
            },
        },
        "variable_types": {
            "prognostic": ["2t", "msl", "10u", "10v"],
            "forcing": ["lsm"],
            "diagnostic": [],
            "target": ["2t", "msl", "10u", "10v"],
        },
        "timesteps": {
            "timestep": "6h",
            "input_relative_date_indices": [-1, 0],
            "output_relative_date_indices": [1],
            "relative_date_indices_training": [-1, 0, 1],
        },
        "shapes": {
            "variables": 5,
            "input_timesteps": 2,
            "ensemble": 1,
            "grid": 40320,
        },
    }


@pytest.fixture()
def sample_inference_dict() -> dict:
    """Flat checkpoint dict as written by anemoi-training.

    The dataset entries sit at the top level alongside scalar fields,
    matching the format that :class:`InferenceMetadata` reshapes via its
    ``@model_validator``.

    Returns
    -------
    dict
        Flat inference metadata dict with one dataset (``"era5_1deg"``).
    """
    return {
        "seed": 42,
        "run_id": "train-abc123",
        "task": "medium-range",
        "dataset_names": ["era5_1deg"],
        "era5_1deg": _make_dataset_block("era5_1deg"),
    }


@pytest.fixture()
def sample_v1_dict(sample_inference_dict) -> dict:
    """Full V1 metadata dict ready for :class:`MetadataV1` validation.

    Parameters
    ----------
    sample_inference_dict : dict
        Flat inference dict fixture.

    Returns
    -------
    dict
        Complete V1 dict including ``schema_version``, ``created_at``, and
        permissive sections.
    """
    return {
        "schema_version": "1.0",
        "created_at": "2024-06-01T12:00:00",
        "metadata_inference": sample_inference_dict,
        "config": {
            "model": {"type": "graphtransformer", "num_layers": 16},
            "training": {"lr": 3e-4, "batch_size": 4},
        },
        "training": {
            "global_step": 100_000,
            "epoch": 50,
            "loss": 0.0312,
        },
        "dataset": {
            "name": "era5_1deg",
            "resolution": 1.0,
            "grid_points": 40320,
            "start": "1979-01-01",
            "end": "2022-12-31",
        },
        "environment": {
            "python_version": "3.11.9",
            "pytorch_version": "2.3.0",
            "anemoi_training_version": "0.4.0",
        },
        "provenance": {
            "git_sha": "deadbeef",
            "hostname": "gpu-node-01",
        },
    }


@pytest.fixture()
def sample_metadata_v1(sample_v1_dict) -> MetadataV1:
    """Validated :class:`MetadataV1` instance built from ``sample_v1_dict``.

    Parameters
    ----------
    sample_v1_dict : dict
        Full V1 dict fixture.

    Returns
    -------
    MetadataV1
        Validated metadata instance.
    """
    return MetadataV1.model_validate(sample_v1_dict)


@pytest.fixture()
def empty_ckpt(tmp_path):
    """Factory: create a PyTorch-style empty checkpoint with one top-level dir.

    Returns a callable ``_make(name="model.ckpt")`` that writes a minimal ZIP
    using the file stem as the top-level archive directory (mirroring how
    ``torch.save`` lays out checkpoints).
    """

    def _make(name: str = "model.ckpt"):
        path = tmp_path / name
        stem = path.stem
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{stem}/data.pkl", b"")
        return path

    return _make


@pytest.fixture()
def tmp_checkpoint(tmp_path, sample_metadata_v1) -> "pytest.FixtureRequest":
    """Temporary ZIP checkpoint file containing serialised metadata.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    sample_metadata_v1 : MetadataV1
        Validated metadata fixture.

    Returns
    -------
    pathlib.Path
        Path to the created ``.ckpt`` ZIP file.
    """
    ckpt_path = tmp_path / "model.ckpt"
    metadata_json = json.dumps(sample_metadata_v1.to_dict(), indent=2)
    with zipfile.ZipFile(ckpt_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(METADATA_PATH, metadata_json)
    return ckpt_path


@pytest.fixture()
def multi_dataset_inference_dict() -> dict:
    """Flat checkpoint dict with two datasets.

    Returns
    -------
    dict
        Flat inference dict with ``era5_1deg`` and ``cerra_025deg`` datasets.
    """
    return {
        "seed": 7,
        "run_id": "train-multi-xyz",
        "task": "medium-range",
        "dataset_names": ["era5_1deg", "cerra_025deg"],
        "era5_1deg": _make_dataset_block("era5_1deg", var_offset=0),
        "cerra_025deg": _make_dataset_block("cerra_025deg", var_offset=10),
    }


@pytest.fixture()
def registry_state():
    """Save and restore MetadataRegistry and MetadataMigrator state.

    Use this fixture when tests mutate the global registry or migration
    state to prevent leakage between tests.

    Yields
    ------
    None
        The fixture yields nothing; it simply saves and restores state.
    """
    from anemoi.metadata.migration import MetadataMigrator
    from anemoi.metadata.registry import MetadataRegistry

    # Save current state
    saved_versions = MetadataRegistry._versions.copy()
    saved_sorted = MetadataRegistry._sorted_versions
    saved_migrations = MetadataMigrator._migrations.copy()

    yield

    # Restore state
    MetadataRegistry._versions = saved_versions
    MetadataRegistry._sorted_versions = saved_sorted
    MetadataMigrator._migrations = saved_migrations


@pytest.fixture()
def sample_v0_dict() -> dict:
    """Minimal valid V0 metadata dict for migration testing.

    Returns
    -------
    dict
        V0 metadata dict with the essential structure for v0_to_v1 migration.
    """
    return {
        "schema_version": "0.0",
        "config": {
            "data": {
                "timestep": "6h",
                "forcing": ["lsm"],
                "diagnostic": [],
            },
            "training": {
                "multistep_input": 2,
                "precision": "16-mixed",
            },
        },
        "data_indices": {
            "data": {
                "input": {
                    "full": [0, 1, 2, 3, 4],
                },
                "output": {
                    "full": [0, 1, 2, 3],
                },
            },
            "model": {
                "output": {
                    "full": [0, 1, 2, 3],
                    "prognostic": [0, 1, 2, 3],
                },
            },
        },
        "dataset": {
            "variables": ["2t", "msl", "10u", "10v", "lsm"],
            "shape": [1000, 5, 1, 40320],
            "frequency": "6h",
        },
        "provenance_training": {
            "git_sha": "abc123",
            "hostname": "train-node-01",
        },
    }

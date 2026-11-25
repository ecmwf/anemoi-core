# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Shared fixtures for the anemoi.metadata test suite.

Metadata content comes from the golden files in ``tests/data``, loaded by
``tests/unit/conftest.py``. The fixtures here hold no metadata.
"""

import zipfile
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def empty_ckpt(tmp_path: Path) -> Callable[..., Path]:
    """Factory: create a PyTorch-style empty checkpoint with one top-level dir.

    Returns a callable ``_make(name="model.ckpt")`` that writes a minimal ZIP
    using the file stem as the top-level archive directory (mirroring how
    ``torch.save`` lays out checkpoints).
    """

    def _make(name: str = "model.ckpt") -> Path:
        path = tmp_path / name
        stem = path.stem
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{stem}/data.pkl", b"")
        return path

    return _make


@pytest.fixture
def registry_state() -> Iterator[None]:
    """Isolate the component registry.

    Use this fixture when tests register component adapters, to prevent
    leakage between tests.

    Yields
    ------
    None
        The fixture yields nothing; it saves the registry state with
        ``ComponentRegistry.snapshot`` and restores it afterwards.
    """
    from anemoi.metadata.registry import ComponentRegistry

    state = ComponentRegistry.snapshot()

    yield

    ComponentRegistry.restore(state)

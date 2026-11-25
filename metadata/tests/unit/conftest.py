# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Fixtures for the golden metadata files in ``tests/data``.

Each ``tests/data/<format>.json`` is a trimmed metadata dict in one checkpoint format
(see ``tests/data/README.md``). ``golden_metadata`` runs a test once per file;
``load_golden`` and the ``golden_loader`` fixture load a single file by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data"


def golden_names() -> list[str]:
    """Return the format names (file stems) of all golden files, sorted.

    Returns
    -------
    list[str]
        Names such as ``"f0_ai_models"`` and ``"f6_main"``.
    """
    return sorted(path.stem for path in GOLDEN_DIR.glob("*.json"))


def load_golden(name: str) -> dict[str, Any]:
    """Load one golden metadata file.

    Parameters
    ----------
    name : str
        Format name, the file stem (``"f6_main"`` for ``tests/data/f6_main.json``).

    Returns
    -------
    dict[str, Any]
        A new dict on every call, so tests may mutate it freely.

    Raises
    ------
    KeyError
        If no golden file has that name.
    """
    path = GOLDEN_DIR / f"{name}.json"
    if not path.is_file():
        msg = f"No golden metadata named {name!r}; available: {golden_names()}"
        raise KeyError(msg)
    return json.loads(path.read_text())


@pytest.fixture(params=golden_names(), ids=lambda name: name)
def golden_metadata(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
    """Yield ``(format_name, metadata_dict)`` for every golden file.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Provides the parametrised format name.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The format name and a fresh copy of the metadata dict.
    """
    return request.param, load_golden(request.param)


@pytest.fixture
def golden_loader() -> Callable[[str], dict[str, Any]]:
    """Return :func:`load_golden`, to load one golden file by name inside a test.

    Returns
    -------
    Callable[[str], dict[str, Any]]
        ``load_golden``.
    """
    return load_golden

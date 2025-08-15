# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from anemoi.models.migrations import Migrator
from anemoi.models.migrations import SaveCkpt


@pytest.fixture(scope="module")
def migrator() -> Migrator:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """
    return Migrator.from_path(Path(__file__).parent / "migrations", "migrations")


@pytest.fixture(scope="module")
def old_migrator() -> Migrator:
    """Load the test migrator with migrations from this folder from the first compatibility group only.

    Returns
    -------
    A Migrator instance
    """
    migrator = Migrator.from_path(Path(__file__).parent / "migrations", "migrations")
    migrator._grouped_migrations.pop()
    return migrator


@pytest.fixture(scope="session")
def ckpt_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("ckpts")


@pytest.fixture
def save_ckpt(tmp_path: Path) -> SaveCkpt:
    return SaveCkpt(tmp_path)


@pytest.fixture(scope="session")
def empty_ckpt(ckpt_dir: Path) -> Path:
    return SaveCkpt(ckpt_dir)({}, migrations=[], name="empty.ckpt")


@pytest.fixture(scope="session")
def recent_ckpt(ckpt_dir: Path) -> Path:
    return SaveCkpt(ckpt_dir)(
        {"foo": "foo", "bar": "bar", "test": "baz"},
        migrations=[
            {
                "name": "1751895180_final",
                "metadata": {"versions": {"migration": "1.0.0", "anemoi-models": "0.9.0"}, "final": True},
            }
        ],
        name="recent.ckpt",
    )

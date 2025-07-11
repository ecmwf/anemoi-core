# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import IncompatibleCheckpointException
from anemoi.models.migrations import Migrator


@pytest.fixture
def migrator() -> Migrator:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """
    return Migrator.from_path(Path(__file__).parent / "migrations", "migrations")


def final_rollback(_):
    raise IncompatibleCheckpointException


@pytest.fixture
def recent_ckpt() -> CkptType:
    return {
        "foo": "foo",
        "bar": "bar",
        "test": "baz",
        "migrations": [{"name": "1751895180_final", "rollback": final_rollback}],
    }

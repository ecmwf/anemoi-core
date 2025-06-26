# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

from anemoi.models.migrations import Migrator


def get_test_migrator() -> Migrator:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """
    return Migrator.from_path(Path(__file__).parent, __name__)


__all__ = ["get_test_migrator"]

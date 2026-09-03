# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.migrations.config import Config
from anemoi.utils.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-training": "0.1.0",
    },
)
# <-- END DO NOT CHANGE


def migrate(config: Config) -> Config:
    """Migrate the config.

    Parameters
    ----------
    config : Config
        The config object to migrated.

    Returns
    -------
    Config
        The migrated config.
    """
    config.add_key("a", "test")
    return config

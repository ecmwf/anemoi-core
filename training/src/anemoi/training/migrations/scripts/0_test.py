# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.migrations.migrator import MigrationManifest
from anemoi.training.migrations.migrator import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-training": "%NEXT_ANEMOI_TRAINING_VERSION%",
    },
)
# <-- END DO NOT CHANGE


def migrate(s: MigrationManifest) -> None:
    g = ["a.b", "a.c"]
    s.move(g, "b")
    print("test")

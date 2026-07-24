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

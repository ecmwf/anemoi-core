# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from textwrap import dedent

from anemoi.training.migrations.migrator import ConfigMigrator

here = Path(__file__).parent
migrations = here / "migrations"


def test_sync_migration_one_migration(tmp_path: Path, get_migrator: Callable[[Iterable[Path]], ConfigMigrator]) -> None:
    config_path = tmp_path / "config.yaml"
    original_content = dedent("""\
    migration_state: null
    """)
    config_path.write_text(original_content)

    migrator = get_migrator([migrations / "0_add_a.py"])
    original_config, migrated_config, executed_migrations = migrator.sync(config_path)

    assert len(executed_migrations) == 1
    assert original_config.to_yaml() == original_content
    assert migrated_config.to_yaml() == dedent("""\
    migration_state: abaf11ba
    a: test
    """)


def test_sync_migrations(tmp_path: Path, get_migrator: Callable[[Iterable[Path]], ConfigMigrator]) -> None:
    config_path = tmp_path / "config.yaml"
    original_content = dedent("""\
    migration_state: null
    """)
    config_path.write_text(original_content)

    migrator = get_migrator(
        [
            migrations / "0_add_a.py",
            migrations / "1_add_b.py",
            migrations / "2_move_a.py",
            migrations / "3_remove_b.py",
        ],
    )
    original_config, migrated_config, executed_migrations = migrator.sync(config_path)

    assert len(executed_migrations) == 4
    assert original_config.to_yaml() == original_content
    assert migrated_config.to_yaml() == dedent("""\
    migration_state: ff5a1427
    c: test
    """)

from anemoi.models.commands.migration import in_incorrect_order
from anemoi.models.commands.migration import new_migrations_from_main_branch
from anemoi.models.migrations import MIGRATION_PATH


def test_in_incorrect_order_correct_order():
    all_migrations = ["1", "2", "3", "4"]
    new_migrations = ["4"]
    incorrect, last_name = in_incorrect_order(all_migrations, new_migrations)
    assert len(incorrect) == 0
    assert last_name == "3"


def test_in_incorrect_order_wrong_order_around():
    all_migrations = ["1", "2", "3"]  # should be 2 1 3
    new_migrations = ["1", "3"]
    incorrect, last_name = in_incorrect_order(all_migrations, new_migrations)
    assert len(incorrect) == 1
    assert incorrect[0] == "1"
    assert last_name == "2"


def test_in_incorrect_order_wrong_order_between():
    all_migrations = ["1", "2", "3", "4"]  # should be 1 4 2 3
    new_migrations = ["2", "3"]
    incorrect, last_name = in_incorrect_order(all_migrations, new_migrations)
    assert len(incorrect) == 2
    assert incorrect[0] == "2"
    assert incorrect[1] == "3"
    assert last_name == "4"


def test_migration_order():
    new_migrations = new_migrations_from_main_branch()
    all_migrations = sorted(
        [file.name for file in MIGRATION_PATH.iterdir() if file.is_file() and file.name != "__init__.py"]
    )
    incorrect_names, _ = in_incorrect_order(all_migrations, new_migrations)
    error_message = "New migrations were added in the main branch. Use `anemoi-models migration fix-order` to update the name of your scripts."
    assert len(incorrect_names) == 0, error_message

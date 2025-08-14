import subprocess
from pathlib import Path

from anemoi.models.migrations import MIGRATION_PATH

here = Path(__file__).parent
root_folder = here.parent.parent.parent


def in_incorrect_order(all_migrations: list[str], new_migrations: list[str]) -> tuple[list[str], str | None]:
    """Tests whether the order of the new migrations is correct.
    All new migrations should be at the end of all_migrations.

    Parameters
    ----------
    all_migrations : list[str]
        All migrations currently in anemoi-models
    new_migrations : list[str]
        New migrations from this PR.

    Returns
    -------
    tuple[list[str], str | None]
        * the list of name in incorrect order
        * the name of the last migration in main
    """
    stop_new = False
    incorrect_order: list[str] = []
    last_name: str | None = None

    for name in reversed(all_migrations):
        if name not in new_migrations and not stop_new:
            stop_new = True
            last_name = name
        elif stop_new and name in new_migrations:
            incorrect_order.append(name)
    return list(reversed(incorrect_order)), last_name


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
    # only addition diffs between the latest commit in origin/main and HEAD
    run_new_migrations = subprocess.run(
        [
            "git diff --name-only --diff-filter=A "
            '$(git log -n 1 origin/main --pretty=format:"%H") '
            f"HEAD {MIGRATION_PATH.resolve()}"
        ],
        capture_output=True,
        shell=True,
    )
    new_migrations = [root_folder / file for file in run_new_migrations.stdout.decode("utf-8").split("\n")]
    new_migrations = [file.name for file in new_migrations if file.is_file() and file.name != "__init__.py"]
    all_migrations = sorted(
        [file.name for file in MIGRATION_PATH.iterdir() if file.is_file() and file.name != "__init__.py"]
    )
    incorrect_names, last_upstream_name = in_incorrect_order(all_migrations, new_migrations)
    error_message = f"New migrations were added in the main branch. You need to rename these migrations: {', '.join(incorrect_names)} to something after {last_upstream_name}."
    assert len(incorrect_names) == 0, error_message

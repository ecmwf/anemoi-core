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

from pathlib import Path
from typing import List
from typing import Tuple

from anemoi.models.migrations import Migration
from anemoi.models.migrations import migrations_from_path

here = Path(__file__)


def load_test_migrations() -> Tuple[List[Migration], List[str]]:
    return migrations_from_path(here.parent, __name__)


__all__ = ["load_test_migrations"]

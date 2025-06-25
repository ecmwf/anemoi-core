from anemoi.models.migrations import CkptType

version = "1.0.0"


def migrate(ckpt: CkptType) -> CkptType:
    """
    This is a first test migration!
    This does nothing, but shows an example migration.
    """
    return ckpt

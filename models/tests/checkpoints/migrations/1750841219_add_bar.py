from anemoi.models.migrations import CkptType

version = "1.0.0"


def migrate(ckpt: CkptType) -> CkptType:
    # Migrate the checkpoint
    assert "bar" not in ckpt
    ckpt["bar"] = "bar"
    return ckpt

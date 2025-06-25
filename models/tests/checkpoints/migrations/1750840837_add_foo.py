from anemoi.models.migrations import CkptType

version = "1.0.0"


def migrate(ckpt: CkptType) -> CkptType:
    # Migrate the checkpoint
    assert "foo" not in ckpt
    ckpt["foo"] = "foo"
    return ckpt

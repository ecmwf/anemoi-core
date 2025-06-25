from anemoi.models.migrations import CkptType
from anemoi.models.migrations import Versions

versions: Versions = {
    "migration": "1.0.0",
    "anemoi-models": "0.8.1.post1",
}


def upgrade(ckpt: CkptType) -> CkptType:
    """Migrate the model"""
    assert "bar" not in ckpt
    ckpt["bar"] = "bar"
    return ckpt


def downgrade(ckpt: CkptType) -> CkptType:
    """Cancels the upgrade function"""
    del ckpt["bar"]
    return ckpt

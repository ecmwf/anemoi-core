# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from anemoi.training.checkpoint.lineage import record_checkpoint_lineage


class ParentUUIDCallback(Callback):
    """A callback that appends a warm-start checkpoint's parent uuid to the module metadata.

    Recorded when Lightning loads the checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        del trainer  # unused
        record_checkpoint_lineage(pl_module, checkpoint, "warm_start")

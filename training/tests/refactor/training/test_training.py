import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.training.data.datamodule import AnemoiMultipleDatasetsDataModule


class FakeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # Example test/assertions
        assert isinstance(batch, dict), "Batch should be a dict"
        raise ValueError(f"First batch returned: {batch}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def test_main(new_config: DictConfig):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None)
    trainer = pl.Trainer()
    trainer.fit(FakeModel(), datamodule=datamodule)


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.refactor.draft import Context
from anemoi.training.data.refactor.draft import sample_provider_factory
from anemoi.training.data.refactor.multiple_datasets_datamodule import AnemoiMultipleDatasetsDataModule
from anemoi.training.data.refactor.read_config import get_data_config_dict
from anemoi.training.data.refactor.read_config import get_sample_config_dict


def test_sampleprovider(new_config: DictConfig):
    dhs_config = get_data_config_dict(new_config.data.data_handlers, "obs")
    sample_config = get_sample_config_dict(new_config.model.sample, "obs")

    context = Context("training", sources=dhs_config, start=2019, end=2020)

    sample_provider = sample_provider_factory(context=context, **sample_config)
    sample_provider[0]
    assert hasattr(sample_provider, "latitudes")
    assert hasattr(sample_provider, "longitudes")
    assert hasattr(sample_provider, "timedeltas")
    assert hasattr(sample_provider, "processors")

    processors = sample_provider.processors(0)
    assert set(processors.keys()) == {"input", "target"}
    assert isinstance(processors["input"]["era5"], list)
    assert len(processors["input"]["era5"][0]) == 2
    assert isinstance(processors["input"]["amsr_h180"], list)
    assert len(processors["input"]["amsr_h180"]) == 0  # no processors

    num_channels = sample_provider.num_channels(0)  # {"input": {"era": (4, 4), "amsr": (1,) }}
    assert set(num_channels.keys()) == {"input", "target"}
    assert set(num_channels["input"].keys()) == {"era5", "amsr_h180"}
    assert isinstance(num_channels["input"]["era5"], int)
    assert isinstance(num_channels["input"]["amsr_h180"], int)
    assert set(num_channels["target"].keys()) == {"era5", "amsr_h180"}
    assert isinstance(num_channels["target"]["era5"], int)
    assert isinstance(num_channels["target"]["amsr_h180"], int)


@pytest.mark.parametrize("which", ["downscaling", "obs", "multiple"])
def test_datamodule(new_config: DictConfig, which: str):
    datamodule = AnemoiMultipleDatasetsDataModule(new_config, None, which=which)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loaders = datamodule.train_dataloader()

    assert isinstance(train_loaders, torch.utils.data.DataLoader)

    batch = next(iter(train_loaders))

    assert isinstance(batch, dict)

    assert set(batch.keys()) == {"input", "target"}

    if which == "obs":
        assert set(batch["input"].keys()) == {"era5", "amsr_h180"}
        assert set(batch["target"].keys()) == {"era5", "amsr_h180"}

        assert batch["input"]["era5"].shape == (1, 2, 7, 1, 40320)  # (, time, vars, ens, latlon)
        assert len(batch["input"]["amsr_h180"]) == 1
        assert batch["input"]["amsr_h180"][0].shape == (1, 3, 187186)
        assert len(batch["target"]["era5"]) == 1
        assert len(batch["target"]["amsr_h180"]) == 1
    if which in ["multiple", "downscaling"]:
        assert set(batch["input"].keys()) == {"era5", "cerra"}
        assert set(batch["target"].keys()) == {"cerra"}


if __name__ == "__main__":
    pytest.main([__file__])

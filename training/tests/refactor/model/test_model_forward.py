import torch
from omegaconf import DictConfig

from anemoi.models.models import AnemoiModelEncProcDec

batch_size = 2
grid_size = 123


def fake_data(config: dict, key: str) -> dict[str, torch.Tensor]:
    assert key in ["input", "output"]
    data_handlers = config.model.model[key]
    sample = {}
    for data_name, vars in data_handlers.items():
        sample[data_name] = torch.rand((batch_size, 1, grid_size, len(vars)))
    return sample


def test_model_forward(new_config: DictConfig):
    input_sample = fake_data(new_config, "input")
    output_sample = fake_data(new_config, "output")

    # Instantiate the model
    model = AnemoiModelEncProcDec(
        new_config,
    )

    pred_sample = model(input_sample)

    assert isinstance(pred_sample, dict)
    assert set(pred_sample.keys()) == set(output_sample.keys())
    for key in pred_sample.keys():
        assert pred_sample[key].shape == output_sample[key].shape

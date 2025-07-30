import pytest

from anemoi.training.data.refactor.read_config import get_config_dict
from anemoi.training.data.refactor.read_config import get_example


@pytest.mark.parametrize("which", ["downscaling"])
def test_sample_config(downscaling_config, which: str):
    sample_config = get_config_dict(downscaling_config)
    expected_config = get_example(which)
    assert sample_config == expected_config


if __name__ == "__main__":
    pytest.main([__file__])

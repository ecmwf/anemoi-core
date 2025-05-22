# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest

from anemoi.models.layers.processor import BaseProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class ProcessorInit:
    num_layers: int = 4
    num_channels: int = 128
    num_chunks: int = 2
    layer_kernels: field(default_factory=DotDict) = None
    cpu_offload: bool = False

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


@pytest.fixture
def processor_init():
    return ProcessorInit()


@pytest.fixture()
def base_processor(processor_init):
    return BaseProcessor(
        **asdict(processor_init),
    )


def test_base_processor_init(processor_init, base_processor):

    assert isinstance(base_processor.num_chunks, int), "num_layers should be an integer"
    assert isinstance(base_processor.num_channels, int), "num_channels should be an integer"

    assert (
        base_processor.num_chunks == processor_init.num_chunks
    ), f"num_chunks ({base_processor.num_chunks}) should be equal to the input num_chunks ({processor_init.num_chunks})"
    assert (
        base_processor.num_channels == processor_init.num_channels
    ), f"num_channels ({base_processor.num_channels}) should be equal to the input num_channels ({processor_init.num_channels})"
    assert (
        base_processor.chunk_size == processor_init.num_layers // processor_init.num_chunks
    ), f"chunk_size ({base_processor.chunk_size}) should be equal to num_layers // num_chunks ({processor_init.num_layers // processor_init.num_chunks})"

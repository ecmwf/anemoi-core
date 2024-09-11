# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.layers.processor import BaseProcessor


@pytest.fixture
def processor_init():
    num_layers = 4
    num_channels = 128
    num_chunks = 2
    activation = "GELU"
    cpu_offload = False
    return num_layers, num_channels, num_chunks, activation, cpu_offload


@pytest.fixture()
def base_processor(processor_init):
    num_layers, num_channels, num_chunks, activation, cpu_offload = processor_init
    return BaseProcessor(
        num_layers,
        num_channels=num_channels,
        num_chunks=num_chunks,
        activation=activation,
        cpu_offload=cpu_offload,
    )


def test_base_processor_init(processor_init, base_processor):
    num_layers, num_channels, num_chunks, *_ = processor_init

    assert isinstance(base_processor.num_chunks, int), "num_layers should be an integer"
    assert isinstance(base_processor.num_channels, int), "num_channels should be an integer"

    assert (
        base_processor.num_chunks == num_chunks
    ), f"num_chunks ({base_processor.num_chunks}) should be equal to the input num_chunks ({num_chunks})"
    assert (
        base_processor.num_channels == num_channels
    ), f"num_channels ({base_processor.num_channels}) should be equal to the input num_channels ({num_channels})"
    assert (
        base_processor.chunk_size == num_layers // num_chunks
    ), f"chunk_size ({base_processor.chunk_size}) should be equal to num_layers // num_chunks ({num_layers // num_chunks})"

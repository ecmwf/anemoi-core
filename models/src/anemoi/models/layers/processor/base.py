# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.utils.checkpoint import checkpoint

from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


class BaseProcessor(nn.Module, ABC):
    """Base Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseProcessor."""
        super().__init__()

        # Each Processor divides the layers into chunks that get assigned to each ProcessorChunk
        self.num_chunks = num_chunks
        self.num_channels = num_channels
        self.chunk_size = num_layers // num_chunks

        self.layer_factory = load_layer_kernels(layer_kernels)

        assert (
            num_layers % num_chunks == 0
        ), f"Number of processor layers ({num_layers}) has to be divisible by the number of processor chunks ({num_chunks})."

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def build_layers(self, processor_chunk_class, *args, **kwargs) -> None:
        """Build Layers."""
        self.proc = nn.ModuleList(
            [
                processor_chunk_class(
                    *args,
                    **kwargs,
                )
                for _ in range(self.num_chunks)
            ],
        )

    def run_layers(self, data: tuple, *args, **kwargs) -> Tensor:
        """Run Layers with checkpoint."""
        for layer in self.proc:
            data = checkpoint(layer, *data, *args, **kwargs, use_reentrant=False)
        return data

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Example forward pass."""
        x = self.run_layers((x,), *args, **kwargs)
        return x

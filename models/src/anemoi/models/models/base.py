# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import TYPE_CHECKING
from typing import Optional

from torch import nn

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.utils.config import DotDict


class AnemoiModel(nn.Module):
    def __init__(self, model_config, input_metadata, output_metadata, metadata):
        super().__init__()
        self.id = str(uuid.uuid4())

        model_config = DotDict(model_config)

        self.model_config = model_config
        self.input_metadata = input_metadata
        self.output_metadata = output_metadata
        self.input_metadata.freeze()
        self.output_metadata.freeze()
        self.metadata = metadata

    def predict_step(
        self,
        batch,
        model_comm_group: Optional["ProcessGroup"] = None,
        gather_out: bool = True,
        **kwargs,
    ):
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.
        model_comm_group : Optional[ProcessGroup], optional
            model communication group, specifies which GPUs work together
        gather_out : str | None, optional
            Specifies how to gather the output, by default None.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """

        # TODO: this method is only used for inference (unpickle + run predict_step)
        # this method should be moved to the model where things are defined.
        # This should be done along with the integration with inference

        with torch.no_grad():
            batch = self.pre_processors(batch, in_place=False)

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, horizonal space, variables
            x = batch[:, 0 : self.multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            y_hat = self(x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs)

            y_hat = self.post_processors(y_hat, in_place=False)

            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(y_hat, -2, apply_shard_shapes(y_hat, -2, grid_shard_shapes), model_comm_group)

        return y_hat

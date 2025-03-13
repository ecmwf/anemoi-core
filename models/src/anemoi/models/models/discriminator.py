import logging
from typing import Optional

import einops
import hydra
import torch
from anemoi.training.utils.debug_hydra import instantiate_debug
from anemoi.utils.config import DotDict
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData
from typing_extensions import override

from anemoi.models.layers.attention import BlockMaskManager
from anemoi.models.layers.embedding import MLPEmbedding
from anemoi.models.models.mixins import AnemoiDiscriminatorModelMixin
from anemoi.models.distributed.shapes import get_shape_shards

LOGGER = logging.getLogger(__name__)
# TODO (rilwan-ade): Currently the processor has not been implemented to have any form of temporal attention/compression in temporal space: This would probably require adaptations or a new encoding/decoding block that has temporal compression as well as spatial compression

# from vector_quantize_pytorch import VectorQuantizer


class TransformerDiscriminator(nn.Module, AnemoiDiscriminatorModelMixin):
    """Discriminator with Transformer architecture."""

    def __init__(
        self,
        config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
        freeze_parameters: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        If using hierarchical VAE then the encoder input and deocder output layer can be different, but there is a specific assumption on symmetry of grids

        Parameters
        ----------
        config : DotDict
            Configuration of the model.
        data_indices : dict
            Data indices.
        graph_data : HeteroData
            Graph data for processing.
        """

        super().__init__()

        self._graph_data = graph_data

        self.graph_name: list = config.graph.decoder[-1]

        self.multi_step = config.training.multistep_input

        # Calculate shapes and indices
        self._calculate_shapes_and_indices(
            data_indices, num_output_channels=config.model_discriminator.num_output_channels
        )
        self._assert_matching_indices(data_indices)

        self._define_tensor_sizes(config)
        # Create trainable tensors
        self._create_trainable_attributes()

        self._register_latlon()

        # Weather State Embedding
        self.weather_state_embedding = MLPEmbedding(
            input_dim=self.num_input_channels,
            hidden_dim=config.model_discriminator.weather_state_embedding_hidden_dim,
            output_dim=config.model_discriminator.weather_state_embedding_out_features,
            initial_bias=False,
        )

        # The base size for attention windows
        # If hierarchical VAE used then we need to make sure that the base window size is a multiple of the grid sizes at each level

        self.initialise_processor(config)

        self.initialise_rope_matrices(config)

        self.classifier = torch.nn.Sequential(
            *[
                torch.nn.Linear(config.model_discriminator.processor.num_channels, self.num_output_channels, bias=True),
            ]
        )  # Using more than one output channels probably encourages the model to judge realism based on different features of the weather

    @override
    def initialise_processor(self, config):
        pass

    def forward(
        self,
        x: Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        sample_from_latent_space: bool = True,
        return_features: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, t, e, grid, _ = x.shape

        effective_batch_size = b * t * e

        x_src_latent_weather = self.weather_state_embedding(
            einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) (vars)")
        )

        # Encode
        x_encoded_weather, features = self.proc(
            x_src_latent_weather,
            model_comm_group=model_comm_group,
            effective_batch_size=effective_batch_size,
            return_features=return_features,
        )

        # NOTE: Check to see if the attention occurs across ensemble members too

        x_encoded_weather = einops.rearrange(
            x_encoded_weather,
            "(batch time ensemble grid) (vars) -> batch time ensemble grid vars",
            batch=b,
            time=t,
            ensemble=e,
            grid=grid,
        )

        x_classified = self.classifier(
            x_encoded_weather
        )  # shape (batch_size, time, ensemble, grid, num_output_channels)

        if return_features:
            features = [
                einops.rearrange(
                    feature,
                    "(batch time ensemble grid) (vars) -> batch time ensemble grid vars",
                    batch=b,
                    time=t,
                    ensemble=e,
                    grid=grid,
                )
                for feature in features
            ]
            return {"x_classified": x_classified, "features": features}
        else:
            return {
                "x_classified": x_classified,
            }

    def proc(
        self,
        x: Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        effective_batch_size: int = None,
        return_features: bool = False,
    ) -> Tensor:
        """Encodes the input tensor into a latent representation."""

        shard_shapes_x = get_shape_shards(x, 0, model_comm_group)
        shard_shapes_noise = None

        if not return_features:
            x = self.processor(
                x,
                batch_size=effective_batch_size,
                model_comm_group=model_comm_group,
                shard_shapes=(shard_shapes_x, shard_shapes_noise),
                return_features=return_features,
            )
            features = None
        else:
            x, features = self.processor(
                x,
                batch_size=effective_batch_size,
                model_comm_group=model_comm_group,
                shard_shapes=(shard_shapes_x, shard_shapes_noise),
                return_features=return_features,
            )

        return x, features

class TransformerDiscriminator_GraphTransformer(TransformerDiscriminator):

    def initialise_processor(self, config: DotDict):

        self.processor = instantiate_debug(
            config.model_discriminator.processor,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )
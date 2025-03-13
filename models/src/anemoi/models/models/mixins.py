import torch
from typing import Dict, Any, Optional
from torch.distributed.distributed_c10d import ProcessGroup
from anemoi.utils.config import DotDict
from anemoi.models.layers.trainable import TrainableTensor
from torch import nn
from anemoi.models.layers.graph import NamedNodesAttributes

class AnemoiReconstructionModelMixin:
    """Mixin class for reconstruction models."""

    def _calculate_shapes_and_indices(self, data_indices: Dict[str, Any]) -> None:
        """Calculate shapes and indices for the model.
        
        Parameters
        ----------
        data_indices : Dict[str, Any]
            Data indices
        """
        self._input_grid_name = data_indices.internal_data.input.grid_name
        self._output_grid_name = data_indices.internal_data.output.grid_name
        
        # Input/output channels
        self.num_input_channels = len(data_indices.internal_data.input.full)
        self.num_output_channels = len(data_indices.internal_data.output.full)
        
        # Grid sizes
        self._input_grid_size = data_indices.internal_data.input.grid_size
        self._output_grid_size = data_indices.internal_data.output.grid_size
        
        # Initialize lists for encoder/decoder grid sizes
        self._list_hidden_grid_size_encoder = []
        self._list_hidden_grid_size_decoder = []
        
    def _assert_matching_indices(self, data_indices: Dict[str, Any]) -> None:
        """Ensure indices match expectations.
        
        Parameters
        ----------
        data_indices : Dict[str, Any]
            Data indices
        """
        # Implementation depends on your specific requirements
        pass
        
    def _define_tensor_sizes(self, config: Dict[str, Any]) -> None:
        """Define tensor sizes for the model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Model configuration
        """
        # Implementation depends on model specifics
        pass
        
    def _create_trainable_attributes(self) -> None:
        """Create trainable attributes for the model."""
        # Implementation depends on model specifics
        pass
        
    def _register_latlon(self) -> None:
        """Register latitude/longitude data."""
        # Implementation depends on model specifics
        pass
        
    def freeze_params(self) -> None:
        """Freeze model parameters."""
        for param in self.parameters():
            param.requires_grad = False


class AnemoiLatentForecastingMixin:

    def _load_vae_from_checkpoint(self, vae_config: DotDict):
        if vae_config.checkpoint_path:
            self.vae_interface = torch.load(vae_config.checkpoint_path, weights_only=False)
            if vae_config.freeze_parameters:
                self.vae_interface.model.freeze_params()
                self.vae_interface.training = False
                self.vae_interface.model.training = False

    def _check_vae_compatibility(self):
        # Check hidden grid of vae matches hidden grid in config
        assert self._graph_name_hidden == self.vae_interface.model._latent_grid_name
        assert (
            self._graph_name_outp == self.vae_interface.model._input_grid_name
        )  # NOTE: This currently assumes our VQVAE's are all symmetric
        assert self._graph_name_inp == self.vae_interface.model._input_grid_name

    def _calculate_shapes_and_indices(self, data_indices: DotDict) -> None:
        self.num_input_channels = self.vae_interface.model.num_input_channels
        self.num_output_channels = self.vae_interface.model.num_output_channels
        self._internal_input_idx = self.vae_interface.model._internal_input_idx
        self._internal_output_idx = self.vae_interface.model._internal_output_idx

    def _assert_matching_indices(self, data_indices: DotDict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _define_tensor_sizes(self, config: DotDict) -> None:
        # self._inp_grid_size = self._graph_data[self._graph_name_inp].num_nodes

        self._inp_grid_size = self.vae_interface._data_grid_size
        self._list_hidden_grid_size = self.vae_interface._list_hidden_grid_size

        # self.trainable_inp_size = config.model.trainable_parameters.data

        self.trainable_inp_size = self.vae_interface.trainable_inp_size
        # self.list_trainable_hidden_size = [val for val in config.model.trainable_parameters.nodes]

        self._list_hidden_grid_size_encoder = self.vae_interface._list_hidden_grid_size_encoder
        self._list_hidden_grid_size_decoder = self.vae_interface._list_hidden_grid_size_decoder

        self.map_gridname_trainablesize = self.vae_interface.map_gridname_trainablesize

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.list_trainable_hidden = nn.ModuleList(
            [
                TrainableTensor(trainable_size=self.trainable_hidden_size, tensor_size=self._hidden_grid_size)
                for _ in self._list_hidden_grid_size
            ]
        )

        self.list_trainable_hidden = self.vae_interface.model.list_trainable_hidden

    def _register_latlon(self) -> None:
        """Register lat/lon buffers.

        The VAE model has latlons registered as attributes in the format latlon_{layer_name}
        These are assigned to this class as well by copying from the VAE interface
        """

        # The self.vae_interface.model has latlons registered as attributes in the format self.latlons_{layer_name}
        # Copy these attributes to this class
        for attr_name in (k for k in self.vae_interface.model.__dict__.keys() if k.startswith("latlons_")):
            setattr(self, attr_name, getattr(self.vae_interface.model, attr_name))


class AnemoiDiscriminatorModelMixin:
    def _calculate_shapes_and_indices(self, data_indices: DotDict, num_output_channels: int) -> None:
        self.num_input_channels = len(data_indices.internal_model.output)
        self.num_output_channels = num_output_channels
        self._internal_input_idx = data_indices.internal_model.output.full
        self._internal_output_idx = []

    def _assert_matching_indices(self, data_indices: DotDict) -> None:
        pass

    def _define_tensor_sizes(self, config: DotDict) -> None:

        self.map_gridname_trainablesize = {
            graph_name: config.model.learned_base_query_dim[graph_name] for graph_name in [self.graph_name]
        }

        self._grid_name = self.graph_name
        self._grid_size = self._graph_data[self._grid_name].num_nodes

    def _register_latlon(
        self,
    ) -> None:
        """Register lat/lon buffers.

        Creates a sin/cos encoding of the lat/lon coordinates of the nodes for the grids

        Parameters
        ----------
        name : str
            Name to store the lat-lon coordinates of the nodes.
        nodes : str
            Name of nodes to map
        """

        # TODO: IN THE FUTURE MAKE IT SUCH THAT ALL THESE EMBEDDINGS ARE ALL ENCODED BY THE SAME LATLON_ENCODER

        # TODO: In the future, get this from the other model being trained instead of re-init 
        
        coords = self._graph_data[self._grid_name].x
        sin_cos_coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        self.register_buffer(f"latlons_{self._grid_name}", sin_cos_coords, persistent=True)

        
    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""

        # grid_names = sorted(list(set(self.list_graph_name_encoder + self.list_graph_name_decoder)))
        from collections import OrderedDict

        self.trainable_latlons_embedding = nn.ModuleDict(
            OrderedDict(
                [
                    (
                        grid,
                        TrainableTensor(
                            trainable_size=self.map_gridname_trainablesize[grid],
                            tensor_size=self._graph_data[grid].num_nodes,
                        ),
                    )
                    for grid in [self.graph_name]
                ]
            )
        )

        # self.trainable_latlons_embedding = NamedNodesAttributes(
            
        # )

    def freeze_params(self):
        """Freezes encoder, decoder, and vector quantizer modules."""

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_params(self):
        """Unfreezes encoder, decoder, and vector quantizer modules."""
        for param in self.parameters():
            param.requires_grad = True



import logging
from typing import Optional

import einops
import torch
from anemoi.training.utils.debug_hydra import instantiate_debug
from anemoi.utils.config import DotDict
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData
from typing_extensions import override

from anemoi.models.layers.attention import BlockMaskManager
from anemoi.models.layers.attention import calculate_scaled_attention_attention_spans
from anemoi.models.layers.embedding import MLPEmbedding
from anemoi.models.models.mixins import AnemoiReconstructionModelMixin


class AnemoiVQVAE(nn.Module, AnemoiReconstructionModelMixin):
    """Vector Quantized Variational Autoencoder."""

    latent_representation_format = "discrete"

    def __init__(
        self,
        config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
        freeze_parameters: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initializes the VQVAE model.

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

        self.list_graph_name_encoder: list = config.graph.encoder
        self.list_graph_name_decoder: list = config.graph.decoder

        self.no_levels_encoder: int = config.model.levels.encoder
        self.no_levels_decoder: int = config.model.levels.decoder

        self.num_channels_encoder: list = config.model.num_channels_encoder
        self.num_channels_decoder: list = config.model.num_channels_decoder

        self.hidden_dim_encoder: list = config.model.hidden_dim_encoder
        self.hidden_dim_decoder: list = config.model.hidden_dim_decoder

        # Checks related to latent levels
        assert (
            len(self.num_channels_encoder) == self.no_levels_encoder
        ), f"Encoder has {len(self.num_channels_encoder)} levels but {self.no_levels_encoder} are expected"  # The encoder includes the qunatization layer?? that goes across
        assert (
            len(self.num_channels_decoder) == self.no_levels_decoder
        ), f"Decoder has {len(self.num_channels_decoder)} levels but {self.no_levels_decoder} are expected"

        assert len(self.list_graph_name_encoder) == self.no_levels_encoder + 1
        assert len(self.list_graph_name_decoder) == self.no_levels_decoder + 1

        self.multi_step = config.training.multistep_input

        # Calculate shapes and indices
        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self._define_tensor_sizes(config)
        # Create trainable tensors
        self._create_trainable_attributes()

        self._register_latlon()

        # NOTE: (rilwan-ade) when not folding time this line has to be changed
        input_dim = (
            self.num_input_channels
            + self.map_gridname_trainablesize[self.list_graph_name_encoder[0]]
            + getattr(self, f"latlons_{self.list_graph_name_encoder[0]}").shape[1]
        )

        # Vector Quantizer
        self._instantiate_vector_quantizer(config)
        # self.latent_space_dim = self.vector_quantizer.vq.dim
        self.latent_space_dim = (
            self.vector_quantizer.vq.dim
        )  # NOTE this is the projected dimension of the latent space, not the actual codebook dim. Update this

        encoder_modules = []
        for i in range(self.no_levels_encoder):

            dst_grid_name = self.list_graph_name_encoder[i + 1]
            src_grid_name = self.list_graph_name_encoder[i]

            sub_graph_processor = graph_data[(dst_grid_name, "to", dst_grid_name)]
            attention_span = calculate_scaled_attention_attention_spans(
                config.model.encoder.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )  # This is usually nodes so scaling method should be down

            encoder_modules.append(
                # NOTE: Below method used for debugging purposes
                # instantiate(
                instantiate_debug(
                    config.model.encoder,
                    #     _recursive_=False,
                    #     _convert_=None,
                    in_channels_src=input_dim if i == 0 else self.num_channels_encoder[i - 1],
                    in_channels_dst=self.map_gridname_trainablesize[dst_grid_name]
                    + getattr(self, f"latlons_{dst_grid_name}").shape[1],
                    hidden_dim=self.hidden_dim_encoder[i],  # Hidden dimension of the mapper and processor
                    out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    sub_graph_mapper=(graph_data[(src_grid_name, "to", dst_grid_name)]),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_encoder[i],
                    dst_grid_size=self._list_hidden_grid_size_encoder[i + 1],
                    sub_graph_processor=sub_graph_processor,
                    ln_autocast=config.model.ln_autocast,
                    attention_span=attention_span,
                    emb_nodes_src_bias=False if i == 0 else True,
                    # cln_noise_dim=self.noise_injector.outp_channels,
                )
            )

        self.encoder = nn.ModuleList(encoder_modules)

        decoder_modules = []
        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]
            src_grid_name = self.list_graph_name_decoder[i]

            decoder_modules.append(
                # instantiate(
                instantiate_debug(
                    config.model.decoder,
                    in_channels_src=self.num_channels_decoder[i - 1] if i != 0 else self.latent_space_dim,
                    in_channels_dst=self.map_gridname_trainablesize[dst_grid_name]
                    + getattr(self, f"latlons_{dst_grid_name}").shape[1],
                    out_channels_dst=self.num_channels_decoder[i],
                    hidden_dim=self.hidden_dim_decoder[i],
                    sub_graph_mapper=(
                        graph_data[
                            (self.list_graph_name_decoder[i], "to", self.list_graph_name_decoder[i + 1])
                        ]  # For some reaosn edge_length is all zeros here
                    ),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_decoder[i],
                    dst_grid_size=(self._list_hidden_grid_size_decoder[i + 1]),
                    sub_graph_processor=graph_data[
                        (self.list_graph_name_decoder[i], "to", self.list_graph_name_decoder[i])
                    ],  # In the decoder we process first then we map
                    ln_autocast=config.model.ln_autocast,
                )
            )

        self.decoder = nn.ModuleList(decoder_modules)

        # self.proj_x_dst_weather = torch.nn.Identity()
        proj_x_dst_weather = []
        proj_x_dst_weather.append(torch.nn.Linear(self.num_channels_decoder[-1], self.num_output_channels, bias=True))

        self.proj_x_dst_weather = torch.nn.Sequential(*proj_x_dst_weather)

        if freeze_parameters:
            self.freeze_params()

    def forward(
        self,
        x: Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        sample_from_latent_space: bool = True,
        detach_encoder_outp: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # TODO: Need to think about how/if the latlon information should be added at each mapping step, currently it is only added at beginning step and then the trainable parameters are used
        batch_size = x.shape[0]
        b, t, e, _, _ = x.shape
        effective_batch_size = b * t * e

        # Encode
        x_encoded = self.encode(x, model_comm_group, effective_batch_size=effective_batch_size)

        # NOTE: Check to see if the attention occurs across ensemble members too

        # Quantize
        x_quantized, encoding_indices, map_loss_breakdown = self.get_latent_representation(
            x_encoded, stochastic_sample_codes=sample_from_latent_space, detach_encoder_outp=detach_encoder_outp
        )

        # Decode
        x_decoded = self.decode(x_quantized, x_encoded, b * t * e, model_comm_group, self.use_quantized)

        x_rec = einops.rearrange(
            x_decoded,
            "(b t e grid) vars -> b t e grid vars",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_decoder[-1],
        )

        x_quantized = einops.rearrange(
            x_quantized,
            "(b t e grid) dim -> b t e grid dim",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_encoder[-1],
        )

        x_latent = einops.rearrange(
            x_encoded,
            "(b t e grid) dim -> b t e grid dim",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_encoder[-1],
        )

        return {
            "x_rec": x_rec,
            "x_quantized": x_quantized,
            "x_latent": x_latent,
            "encoding_indices": encoding_indices,
            "map_loss_breakdown": map_loss_breakdown,
        }

    def encode(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        """Encodes the input tensor into a latent representation."""
        b, t, e, _, _ = x.shape  # batch size, time, ensemble, grid, variables

        x_src_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) (vars)"),
                self.trainable_latlons_embedding[self.list_graph_name_encoder[0]](
                    getattr(self, f"latlons_{self.list_graph_name_encoder[0]}"), batch_size=b * t * e
                ),
            ),
            dim=-1,
        )
        for i in range(self.no_levels_encoder):
            # NOTE (rilwan-ade): should noise be releated to level in hirarchical encoder
            dst_grid_name = (
                self.list_graph_name_encoder[i + 1]
                if i != self.no_levels_encoder - 1
                else self.list_graph_name_decoder[0]
            )

            x_dst_latlon_latent = self.trainable_latlons_embedding[dst_grid_name](
                getattr(self, f"latlons_{dst_grid_name}"), batch_size=b * t * e
            )

            _, x_dst_latent = self.encoder[i](
                x_src_latent,
                x_dst_latlon_latent,
                batch_size=b * t * e,
                model_comm_group=model_comm_group,
            )  # NOTE (rilwan-ade): Important x_dst_latent is updated at each for-loop. x_src_latent may or may not be updated

            # In next encoder block, our source is now the destination grid of the previous encoder block
            x_src_latent = x_dst_latent

        return x_dst_latent

    def get_latent_representation(
        self,
        x_encoded: Tensor,
        sample_codebook_temp=1.0,
        stochastic_sample_codes=False,
        freeze_codebook=False,
        detach_encoder_outp: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Applies vector quantization to the latent representation."""
        # NOTE : check during discriminator update whether or not freeze codebook should be added here
        x_quantized, encoding_indices, map_loss_breakdown = self.vector_quantizer(
            x_encoded,
            sample_codebook_temp=sample_codebook_temp,
            stochastic_sample_codes=stochastic_sample_codes,
            freeze_codebook=freeze_codebook or detach_encoder_outp,
        )

        if detach_encoder_outp:
            x_quantized = x_quantized.detach()
            x_encoded = x_encoded.detach()
            # encoding_indices = encoding_indices.detach()
            # map_loss_breakdown = {k: v.detach() for k, v in map_loss_breakdown.items()}

        return x_quantized, encoding_indices, map_loss_breakdown

    def decode(
        self,
        x_quantized: Tensor,
        x_latent: Tensor,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        use_quantized: bool = True,
    ) -> Tensor:
        """Decodes the quantized latent representation back to the original space."""
        # z_q is x_quantized

        x_src_latent = x_quantized if use_quantized else x_latent

        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]

            x_dst_latlon_latent = self.trainable_latlons_embedding[dst_grid_name](
                getattr(self, f"latlons_{dst_grid_name}"), batch_size=batch_size
            )
            x_dst_latent = self.decoder[i](
                x_src_latent, x_dst_latlon_latent, batch_size=batch_size, model_comm_group=model_comm_group
            )

            x_src_latent = x_dst_latent

        # NOTE: TO be removed/reimplemented later - Currently only used to push all values above 0.1 - when masked weath4er modelling is used for vae
        x_dst_latent = self.proj_x_dst_weather(x_dst_latent)

        return x_dst_latent

    def _instantiate_vector_quantizer(self, config: DotDict):
        # Vector Quantizer
        self.vector_quantizer = instantiate_debug(
            config.model.vector_quantizer,
            # use_ddp = # check if more
        )



class AnemoiBetaVAE(nn.Module, AnemoiReconstructionModelMixin):
    """Vector Quantized Variational Autoencoder."""

    latent_representation_format = "continuous"

    def __init__(
        self,
        config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
        freeze_parameters: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initializes the VQVAE model.

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

        self.list_graph_name_encoder: list = config.graph.encoder
        self.list_graph_name_decoder: list = config.graph.decoder

        self.no_levels_encoder: int = config.model.levels.encoder
        self.no_levels_decoder: int = config.model.levels.decoder

        self.latent_space_dim: int = config.model.latent_space_dim

        self.num_channels_encoder: list = config.model.num_channels_encoder
        self.num_channels_decoder: list = config.model.num_channels_decoder

        self.hidden_dim_encoder: list = config.model.hidden_dim_encoder
        self.hidden_dim_decoder: list = config.model.hidden_dim_decoder

        # Checks related to latent levels
        assert (
            len(self.num_channels_encoder) == self.no_levels_encoder
        ), f"Encoder has {len(self.num_channels_encoder)} levels but {self.no_levels_encoder} are expected"  # The encoder includes the qunatization layer?? that goes across
        assert (
            len(self.num_channels_decoder) == self.no_levels_decoder
        ), f"Decoder has {len(self.num_channels_decoder)} levels but {self.no_levels_decoder} are expected"

        assert len(self.list_graph_name_encoder) == self.no_levels_encoder + 1
        assert len(self.list_graph_name_decoder) == self.no_levels_decoder + 1

        self.multi_step = config.training.multistep_input

        # Calculate shapes and indices
        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self._define_tensor_sizes(config)
        # Create trainable tensors
        self._create_trainable_attributes()

        self._register_latlon()

        # NOTE: (rilwan-ade) when not folding time this line has to be changed
        input_dim = (
            self.num_input_channels
            + self.map_gridname_trainablesize[self.list_graph_name_encoder[0]]
            + getattr(self, f"latlons_{self.list_graph_name_encoder[0]}").shape[1]
        )

        encoder_modules = []
        for i in range(self.no_levels_encoder):

            dst_grid_name = self.list_graph_name_encoder[i + 1]
            src_grid_name = self.list_graph_name_encoder[i]

            sub_graph_processor = graph_data[(dst_grid_name, "to", dst_grid_name)]
            attention_span = calculate_scaled_attention_attention_spans(
                config.model.base_processor_attention_span,
                self._input_grid_name,
                src_grid_name,
                scaling_method="scale_span_relative_to_grid_size",
                _graph_data=self._graph_data,
            )  # This is usually nodes so scaling method should be down

            encoder_modules.append(
                # NOTE: Below method used for debugging purposes
                # instantiate(
                instantiate_debug(
                    config.model.encoder,
                    #     _recursive_=False,
                    #     _convert_=None,
                    in_channels_src=input_dim if i == 0 else self.num_channels_encoder[i - 1],
                    in_channels_dst=self.map_gridname_trainablesize[dst_grid_name]
                    + getattr(self, f"latlons_{dst_grid_name}").shape[1],
                    hidden_dim=self.hidden_dim_encoder[i],  # Hidden dimension of the mapper and processor
                    # out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    out_channels_dst=self.num_channels_encoder[i],  # Output channels of the mapper
                    sub_graph_mapper=(graph_data[(src_grid_name, "to", dst_grid_name)]),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_encoder[i],
                    dst_grid_size=self._list_hidden_grid_size_encoder[i + 1],
                    sub_graph_processor=sub_graph_processor,
                    ln_autocast=config.model.ln_autocast,
                    attention_span=attention_span,
                    emb_nodes_src_bias=False if i == 0 else True,
                )
            )

        self.encoder = nn.ModuleList(encoder_modules)

        decoder_modules = []
        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]
            src_grid_name = self.list_graph_name_decoder[i]

            decoder_modules.append(
                # instantiate(
                instantiate_debug(
                    config.model.decoder,
                    in_channels_src=self.num_channels_decoder[i - 1] if i != 0 else self.latent_space_dim,
                    in_channels_dst=self.map_gridname_trainablesize[dst_grid_name]
                    + getattr(self, f"latlons_{dst_grid_name}").shape[1],
                    out_channels_dst=self.num_channels_decoder[i],
                    hidden_dim=self.hidden_dim_decoder[i],
                    sub_graph_mapper=(
                        graph_data[
                            (self.list_graph_name_decoder[i], "to", self.list_graph_name_decoder[i + 1])
                        ]  # For some reaosn edge_length is all zeros here
                    ),
                    sub_graph_edge_attributes=config.model.attributes.edges,
                    src_grid_size=self._list_hidden_grid_size_decoder[i],
                    dst_grid_size=(self._list_hidden_grid_size_decoder[i + 1]),
                    sub_graph_processor=graph_data[
                        (self.list_graph_name_decoder[i], "to", self.list_graph_name_decoder[i])
                    ],  # In the decoder we process first then we map
                    ln_autocast=config.model.ln_autocast,
                )
            )

        self.decoder = nn.ModuleList(decoder_modules)

        if self.num_channels_encoder[-1] != 2 * self.latent_space_dim:
            self.proj_latent_space = nn.Linear(self.num_channels_encoder[-1], 2 * self.latent_space_dim, bias=False)
        else:
            self.proj_latent_space = torch.nn.Identity()

        # self.proj_x_dst_weather = torch.nn.Identity()
        proj_x_dst_weather = []
        proj_x_dst_weather.append(torch.nn.Linear(self.num_channels_decoder[-1], self.num_output_channels, bias=True))

        self.proj_x_dst_weather = torch.nn.Sequential(*proj_x_dst_weather)

        if freeze_parameters:
            self.freeze_params()

        # Laten Space related transformations

    def get_encoder_named_parameters(self):
        named_parameters = []
        encoder_parameters = self.encoder.named_parameters()

        for _ in encoder_parameters:
            named_parameters.append(_)

        latent_proj = self.proj_latent_space.named_parameters()
        for _ in latent_proj:
            named_parameters.append(_)

        return named_parameters

    def get_decoder_named_parameters(self):
        named_parameters = []
        decoder_parameters = self.decoder.named_parameters()
        for _ in decoder_parameters:
            named_parameters.append(_)

        outp_proj = self.proj_x_dst_weather.named_parameters()
        for _ in outp_proj:
            named_parameters.append(_)

        return named_parameters

    def get_encoder_decoder_shared_named_parameters(self):
        named_parameters = []
        for _ in self.trainable_latlons_embedding.named_parameters():
            named_parameters.append(_)
        return named_parameters

    def encode(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        """Encodes the input tensor into a latent representation."""
        b, t, e, _, _ = x.shape  # batch size, time, ensemble, grid, variables

        x_src_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) (vars)"),
                self.trainable_latlons_embedding[self.list_graph_name_encoder[0]](
                    getattr(self, f"latlons_{self.list_graph_name_encoder[0]}"), batch_size=b * t * e
                ),
            ),
            dim=-1,
        )
        for i in range(self.no_levels_encoder):

            dst_grid_name = (
                self.list_graph_name_encoder[i + 1]
                if i != self.no_levels_encoder - 1
                else self.list_graph_name_decoder[0]
            )

            x_dst_latlon_latent = self.trainable_latlons_embedding[dst_grid_name](
                getattr(self, f"latlons_{dst_grid_name}"), batch_size=b * t * e
            )

            _, x_dst_latent = self.encoder[i](
                x_src_latent,
                x_dst_latlon_latent,
                batch_size=b * t * e,
                model_comm_group=model_comm_group,
            )  # NOTE (rilwan-ade): Important x_dst_latent is updated at each for-loop. x_src_latent may or may not be updated

            # In next encoder block, our source is now the destination grid of the previous encoder block
            x_src_latent = x_dst_latent

        return x_dst_latent

    def get_sample(self, x_encoded: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        x_encoded = self.proj_latent_space(x_encoded)

        x_mu, x_logvar = x_encoded.chunk(2, dim=-1)

        x_sampled = x_mu + torch.randn_like(x_logvar) * torch.exp(x_logvar / 2)
        return x_sampled, x_mu, x_logvar

    def decode(
        self,
        x_sampled: Tensor,
        x_mu: Tensor,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        use_deterministic: bool = True,
    ) -> Tensor:
        """Decodes the quantized latent representation back to the original space."""
        # z_q is x_quantized

        x_src_latent = x_mu if use_deterministic else x_sampled

        for i in range(0, self.no_levels_decoder):
            dst_grid_name = self.list_graph_name_decoder[i + 1]

            x_dst_latlon_latent = self.trainable_latlons_embedding[dst_grid_name](
                getattr(self, f"latlons_{dst_grid_name}"), batch_size=batch_size
            )
            x_dst_latent = self.decoder[i](
                x_src_latent, x_dst_latlon_latent, batch_size=batch_size, model_comm_group=model_comm_group
            )

            x_src_latent = x_dst_latent

        x_dst_latent = self.proj_x_dst_weather(x_dst_latent)

        return x_dst_latent

    def forward(
        self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None, sample_from_latent_space: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        # TODO: Need to think about how/if the latlon information should be added at each mapping step, currently it is only added at beginning step and then the trainable parameters are used
        batch_size = x.shape[0]
        b, t, e, _, _ = x.shape

        # Encode
        x_encoded = self.encode(x, model_comm_group)  # This is essentially x_mu

        # NOTE: Check to see if the attention occurs across ensemble members too

        # Quantize

        x_sampled, x_mu, x_logvar = self.get_sample(x_encoded)

        # Decode
        if sample_from_latent_space:
            x_decoded = self.decode(x_sampled, x_mu, b * t * e, model_comm_group)
        else:
            x_decoded = self.decode(x_mu, x_mu, b * t * e, model_comm_group)

        x_rec = einops.rearrange(
            x_decoded,
            "(b t e grid) vars -> b t e grid vars",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_decoder[-1],
        )

        x_sampled = einops.rearrange(
            x_sampled,
            "(b t e grid) dim -> b t e grid dim",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_encoder[-1],
        )

        x_mu = einops.rearrange(
            x_mu,
            "(b t e grid) dim -> b t e grid dim",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_encoder[-1],
        )

        x_logvar = einops.rearrange(
            x_logvar,
            "(b t e grid) dim -> b t e grid dim",
            b=b,
            t=t,
            e=e,
            grid=self._list_hidden_grid_size_encoder[-1],
        )

        return {
            "x_rec": x_rec,
            "x_quantized": x_sampled,
            "x_latent": x_mu,
            "x_latent_logvar": x_logvar,
        }


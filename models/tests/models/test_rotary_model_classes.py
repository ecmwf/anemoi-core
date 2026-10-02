# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rotary embeddings reach every transformer part of every model class.

Each model is built with rotary embeddings in its processor(s) and transformer mappers. The module
checks that the queries and keys it turns belong to the nodes it was set up for, so a forward and
backward pass through the whole model shows that every part received the coordinates of its own nodes.
"""

import copy

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.spherical_rotary import SphericalRotaryEmbedding
from anemoi.models.models import AnemoiEnsModelEncProcDec
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models import AnemoiModelEncProcDecHierarchical
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec

DATA = ReducedGrid.octahedral(8)
HIDDEN = [ReducedGrid.octahedral(4), ReducedGrid.octahedral(2), ReducedGrid.octahedral(1)]
NUM_CHANNELS = 32
ROTARY = {"max_frequency": 30.0, "backend": "torch"}
TRANSFORMER = {
    "num_chunks": 1,
    "num_heads": 4,
    "mlp_hidden_ratio": 2,
    "window_size": None,
    "dropout_p": 0.0,
    "attention_implementation": "scaled_dot_product_attention",
    "softcap": 0.0,
    "cpu_offload": False,
    "gradient_checkpointing": False,
    "rotary_embeddings": ROTARY,
}


def conditional_kernels(condition_shape: int, with_shape: bool = True) -> dict:
    """Conditional layer norms; the mapper blocks give the width themselves, so theirs leave it out."""
    layer_norm = {
        "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
        "condition_shape": condition_shape,
        "zero_init": False,
        "autocast": False,
    }
    if with_shape:
        layer_norm["normalized_shape"] = NUM_CHANNELS
    return {"LayerNorm": layer_norm}


def mapper(direction: str, layer_kernels: dict) -> dict:
    target = "TransformerForwardMapper" if direction == "forward" else "TransformerBackwardMapper"
    return {
        "_target_": f"anemoi.models.layers.mapper.{target}",
        "num_channels": NUM_CHANNELS,
        "qk_norm": False,
        "layer_kernels": layer_kernels,
        **TRANSFORMER,
    }


def base_config(model_target: str, hidden_nodes_name, processor_kernels=None, mapper_kernels=None) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "num_channels": NUM_CHANNELS,
                "model": {"_target_": model_target, "hidden_nodes_name": hidden_nodes_name, "latent_skip": True},
                "node_trainable_parameters": {"data": 0, **{f"hidden_{i}": 0 for i in range(3)}, "hidden": 0},
                "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.TransformerProcessor",
                    "num_channels": NUM_CHANNELS,
                    "num_layers": 2,
                    "qk_norm": True,
                    "layer_kernels": processor_kernels or {},
                    **TRANSFORMER,
                },
                "encoders": {
                    "0": {
                        "source_datasets": ["data"],
                        "dataset_fusing_strategy": "not_supported",
                        "mapper": mapper("forward", mapper_kernels or {}),
                    }
                },
                "decoders": {
                    "0": {
                        "target_datasets": ["data"],
                        "target_node_features": ["encoded_data"],
                        "mapper": {**mapper("backward", mapper_kernels or {}), "initialise_data_extractor_zero": False},
                    }
                },
                "residual": {
                    "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}}
                },
                "bounding": {"datasets": {"data": []}},
                "output_mask": {"datasets": {"data": {"_target_": "anemoi.training.utils.masks.NoOutputMask"}}},
            }
        }
    )


def graph(hidden_names: list[str]) -> HeteroData:
    data = HeteroData()
    for name, grid in [("data", DATA), *zip(hidden_names, HIDDEN)]:
        data[name].x = grid.coords.float()
        data[name].num_nodes = grid.num_points
    return data


def data_indices() -> dict[str, IndexCollection]:
    config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    return {"data": IndexCollection(config, {"prog0": 0, "prog1": 1, "force": 2, "diag": 3})}


def build(model_class, config: DictConfig, hidden_names: list[str], **extra):
    torch.manual_seed(0)
    return model_class(
        model_config=config.model,
        data_indices=data_indices(),
        statistics={"data": None},
        n_step_input=2,
        n_step_output=1,
        graph_data=graph(hidden_names),
        **extra,
    )


def inputs(ensemble: int = 1) -> dict[str, torch.Tensor]:
    num_vars = len(data_indices()["data"].model.input)
    return {"data": torch.randn(2, 2, ensemble, DATA.num_points, num_vars, generator=torch.Generator().manual_seed(1))}


def attention_layers(model) -> list[MultiHeadSelfAttention]:
    # The mapper blocks replace the self attention layer their parent block builds; only layers in use count.
    return [m for m in model.modules() if isinstance(m, MultiHeadSelfAttention)]


def without_rotary(config: DictConfig) -> DictConfig:
    config = copy.deepcopy(config)
    for part in [config.model.processor, config.model.encoders["0"].mapper, config.model.decoders["0"].mapper]:
        part.rotary_embeddings = None
    for key in ("upscale_mapper", "downscale_mapper"):
        if key in config.model:
            config.model[key].rotary_embeddings = None
    return config


def check_forward_and_backward(model, run) -> torch.Tensor:
    out = run(model)
    out.float().pow(2).mean().backward()
    for name, p in model.named_parameters():
        if ".attention.lin_q" in name or ".attention.lin_k" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    return out


def check_rotary_everywhere(model) -> None:
    layers = attention_layers(model)
    assert layers and all(isinstance(layer.rotary, SphericalRotaryEmbedding) for layer in layers)


# ------------------------------------------------------------------------------------------ the models


def run_deterministic(model):
    return model(inputs())["data"]


def run_ensemble(model):
    return model(inputs(ensemble=2), fcstep=0)["data"]


def run_transport(model):
    x = inputs()
    num_out = len(data_indices()["data"].model.output)
    y_noised = {"data": torch.randn(2, 1, 1, DATA.num_points, num_out, generator=torch.Generator().manual_seed(2))}
    sigma = {"data": torch.full((2, 1, 1, 1, 1), 0.5)}
    return model(x, y_noised, sigma)["data"]


def deterministic_case():
    return (
        AnemoiModelEncProcDec,
        base_config("anemoi.models.models.AnemoiModelEncProcDec", "hidden"),
        ["hidden"],
        {},
        run_deterministic,
    )


def ensemble_case():
    config = base_config(
        "anemoi.models.models.AnemoiEnsModelEncProcDec", "hidden", processor_kernels=conditional_kernels(4)
    )
    config.model.condition_on_residual = False
    config.model.noise_injector = {
        "_target_": "anemoi.models.layers.ensemble.NoiseConditioning",
        "noise_std": 1,
        "noise_channels_dim": 4,
        "noise_mlp_hidden_dim": 8,
        "noise_matrix": None,
        "noise_edges_name": None,
        "edge_weight_attribute": None,
        "row_normalize_noise_matrix": False,
        "autocast": False,
        "layer_kernels": {"Activation": {"_target_": "torch.nn.GELU"}},
    }
    return AnemoiEnsModelEncProcDec, config, ["hidden"], {}, run_ensemble


def transport_case(tendency: bool):
    target = "AnemoiTransportTendModelEncProcDec" if tendency else "AnemoiTransportModelEncProcDec"
    config = base_config(
        f"anemoi.models.models.{target}",
        "hidden",
        processor_kernels=conditional_kernels(16),
        mapper_kernels=conditional_kernels(16, with_shape=False),
    )
    config.model.model.transport = {
        "objective": "edm_diffusion",
        "sigma_data": 1.0,
        "noise_channels": 32,
        "noise_cond_dim": 16,
        "sigma_max": 100.0,
        "sigma_min": 0.02,
        "rho": 7.0,
        "noise_embedder": {
            "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
            "num_channels": 32,
            "max_period": 1000,
        },
    }
    if tendency:
        config.model.condition_on_residual = False
    model_class = AnemoiTransportTendModelEncProcDec if tendency else AnemoiTransportModelEncProcDec
    return model_class, config, ["hidden"], {}, run_transport


def hierarchical_case():
    names = ["hidden_0", "hidden_1", "hidden_2"]
    config = base_config("anemoi.models.models.AnemoiModelEncProcDecHierarchical", names)
    config.model.enable_hierarchical_level_processing = True
    config.model.level_process_num_layers = 1
    # Channels double at every coarser level; the mappers between levels are sized by the model.
    config.model.upscale_mapper = {k: v for k, v in mapper("forward", {}).items() if k != "num_channels"}
    config.model.downscale_mapper = {k: v for k, v in mapper("backward", {}).items() if k != "num_channels"}
    return AnemoiModelEncProcDecHierarchical, config, names, {}, run_deterministic


CASES = {
    "deterministic": deterministic_case,
    "ensemble": ensemble_case,
    "transport": lambda: transport_case(tendency=False),
    "transport_tendency": lambda: transport_case(tendency=True),
    "hierarchical": hierarchical_case,
}


@pytest.mark.parametrize("case", list(CASES))
def test_every_attention_layer_turns_its_own_nodes(case):
    model_class, config, hidden_names, extra, run = CASES[case]()
    model = build(model_class, config, hidden_names, **extra)
    check_rotary_everywhere(model)
    out = check_forward_and_backward(model, run)
    assert torch.isfinite(out).all()

    # The same weights without rotary embeddings give another forecast, so the turn is applied.
    plain = build(model_class, without_rotary(config), hidden_names, **extra)
    plain.load_state_dict(model.state_dict())
    assert all(layer.rotary is None for layer in attention_layers(plain))
    with torch.no_grad():
        assert not torch.allclose(run(plain), run(model))


def test_hierarchical_levels_turn_with_their_own_head_size():
    model_class, config, hidden_names, extra, _ = hierarchical_case()
    model = build(model_class, config, hidden_names, **extra)
    # 32, 64 and 128 channels over 4 heads: 8, 16 and 32 channels per head.
    assert model.processor.proc[0].attention.rotary.head_dim == 32
    level_head_dims = {name: p.proc[0].attention.rotary.head_dim for name, p in model.down_level_processor.items()}
    assert level_head_dims == {"hidden_0": 8, "hidden_1": 16}
    upscale = model.upscale["hidden_0"].proc.attention.rotary
    assert (upscale.query_cos.shape[0], upscale.key_cos.shape[0]) == (HIDDEN[1].num_points, HIDDEN[0].num_points)

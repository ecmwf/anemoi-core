# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import MethodType
from types import SimpleNamespace

import torch

from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec


class DummyNodeAttributes:
    def __init__(self, hidden_nodes: int) -> None:
        self.num_nodes = {"hidden": hidden_nodes}

    def __call__(self, graph_name: str, batch_size: int) -> torch.Tensor:
        assert graph_name == "hidden"
        return torch.zeros(batch_size * self.num_nodes[graph_name], 1)


class RaisingGraphProvider:
    def get_edges(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("decoder graph provider should not be used directly in the ensemble forward path")


class RaisingDecoder:
    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("decoder should not be called directly in the ensemble forward path")


def test_ensemble_forward_uses_optional_refiner_decode_path() -> None:
    model = AnemoiEnsModelEncProcDec.__new__(AnemoiEnsModelEncProcDec)
    torch.nn.Module.__init__(model)

    model._graph_name_data = "data"
    model._graph_name_hidden = "hidden"
    model.n_step_output = 1
    model.node_attributes = {"data": DummyNodeAttributes(hidden_nodes=2)}

    model._get_consistent_dim = lambda x, dim: next(iter(x.values())).shape[dim]
    model._resolve_in_out_sharded = lambda dataset_names, grid_shard_shapes: {
        dataset_name: False for dataset_name in dataset_names
    }
    model._assert_valid_sharding = lambda *args, **kwargs: None

    model.encoder_graph_provider = {"data": SimpleNamespace(get_edges=lambda **kwargs: (None, None, None))}
    model.encoder = {"data": lambda pair, **kwargs: (pair[0], pair[1] + 1.0)}

    model.noise_injector = lambda x, **kwargs: (x, None)
    model.processor_graph_provider = SimpleNamespace(get_edges=lambda **kwargs: (None, None, None))
    model.processor = lambda x, **kwargs: x

    model.decoder_graph_provider = {"data": RaisingGraphProvider()}
    model.decoder = {"data": RaisingDecoder()}

    x_skip = torch.randn(1, 1, 2, 3, 1)
    x_data_latent = torch.randn(6, 2)
    shard_shapes_data = [[6, 2]]
    model._assemble_input = (
        lambda x, fcstep, batch_ens_size, grid_shard_shapes=None, model_comm_group=None, dataset_name=None: (
            x_data_latent,
            x_skip,
            shard_shapes_data,
        )
    )

    helper_out = torch.randn(6, 4)
    decode_calls = []
    assemble_calls = []

    def fake_decode(self, **kwargs):
        decode_calls.append(kwargs)
        return helper_out

    def fake_assemble_output(self, x_out, x_skip_arg, batch_size, batch_ens_size, dtype, dataset_name=None):
        assemble_calls.append(
            {
                "x_out": x_out,
                "x_skip": x_skip_arg,
                "batch_size": batch_size,
                "batch_ens_size": batch_ens_size,
                "dtype": dtype,
                "dataset_name": dataset_name,
            }
        )
        return x_out

    model._decode_with_optional_refiner = MethodType(fake_decode, model)
    model._assemble_output = MethodType(fake_assemble_output, model)

    x = {"data": torch.randn(1, 1, 2, 3, 4)}
    out = model.forward(x, fcstep=5)

    assert len(decode_calls) == 1
    assert decode_calls[0]["dataset_name"] == "data"
    assert decode_calls[0]["batch_size"] == 2
    assert decode_calls[0]["x_latent_proc"].shape == (4, 1)
    assert decode_calls[0]["x_data_latent"] is x_data_latent
    assert decode_calls[0]["shard_shapes_hidden"] == [[4, 1]]
    assert decode_calls[0]["shard_shapes_data"] == shard_shapes_data
    assert decode_calls[0]["in_out_sharded"] is False

    assert len(assemble_calls) == 1
    assert assemble_calls[0]["x_out"] is helper_out
    assert assemble_calls[0]["x_skip"] is x_skip
    assert assemble_calls[0]["batch_size"] == 1
    assert assemble_calls[0]["batch_ens_size"] == 2
    assert assemble_calls[0]["dtype"] == x["data"].dtype
    assert assemble_calls[0]["dataset_name"] == "data"
    assert out["data"] is helper_out


if __name__ == "__main__":
    test_ensemble_forward_uses_optional_refiner_decode_path()

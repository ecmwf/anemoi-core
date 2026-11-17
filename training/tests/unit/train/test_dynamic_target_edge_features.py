# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Synthetic checks, runnable with unittest even without pytest, torchdata or site datasets."""

import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.schemas.models import BaseModelSchema
from anemoi.models.utils.graphs import split_graph_config
from anemoi.training.schemas.data import DataSchema
from anemoi.training.tasks import Autoencoder
from anemoi.utils.config import DotDict

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dynamic_target_edge_features.yaml"


def _indices() -> dict[str, IndexCollection]:
    return {
        "data": IndexCollection(OmegaConf.create({"forcing": ["u"], "diagnostic": []}), {"u": 0}),
        "obs": IndexCollection(
            OmegaConf.create({"forcing": ["level"], "diagnostic": ["TURBIND"]}),
            {"TURBIND": 0, "level": 1},
        ),
    }


def _batch(device: str = "cpu", counts: tuple[int, ...] = (2, 0, 1), target_value: float = 0.5) -> Batch:
    stats = {"mean": [0.0, 100.0], "stdev": [1.0, 20.0], "minimum": [0.0, 0.0], "maximum": [1.0, 200.0]}
    stats = {key: np.asarray(value) for key, value in stats.items()}
    data = []
    for i, count in enumerate(counts):
        sample = torch.empty(count, 2, device=device)
        sample[:, 0] = target_value
        sample[:, 1] = torch.arange(count, device=device) * 20.0 + 100.0 + i * 20.0
        data.append(sample)
    return Batch(
        {
            "data": GriddedSource(
                name="data",
                variables=["u"],
                statistics={},
                layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
                values=torch.ones(len(counts), 1, 1, 2, 1, device=device),
                coordinates=torch.tensor([[0.0, 0.0], [0.001, 0.001]], device=device),
            ),
            "obs": TabularSource(
                name="obs",
                variables=["TURBIND", "level"],
                statistics=stats,
                layout=TensorLayout(grid=0, variables=1),
                values=data,
                coordinates=[torch.arange(n, device=device)[:, None].expand(-1, 2) * 0.0001 for n in counts],
                timedeltas=[torch.zeros(n, device=device) for n in counts],
                boundaries=[(slice(0, n),) for n in counts],
            ),
        },
    )


def _model(
    graph_attention_backend: str = "pyg",
    *,
    model_class: type[AnemoiModelEncProcDec] = AnemoiModelEncProcDec,
    hidden_nodes_name: str | dict[str, str] = "hidden",
    transport_objective: str = "edm_diffusion",
) -> AnemoiModelEncProcDec:
    config = OmegaConf.load(CONFIG_PATH)
    config.model.model.hidden_nodes_name = hidden_nodes_name
    if issubclass(model_class, AnemoiEnsModelEncProcDec):
        config.model.condition_on_residual = False
        config.model.noise_injector = {"_target_": "anemoi.models.layers.ensemble.NoOpNoiseInjector"}
    if issubclass(model_class, AnemoiTransportModelEncProcDec):
        config.model.layer_kernels.LayerNorm = {
            "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
            "condition_shape": 4,
            "zero_init": False,
            "autocast": False,
        }
        config.model.model.transport = {
            "objective": transport_objective,
            "noise_channels": 8,
            "noise_cond_dim": 4,
            "noise_embedder": {
                "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                "num_channels": 8,
            },
        }
    # Supply a tiny static graph in memory; exercise real mappers and processor.
    graph = HeteroData()
    for name in ("data", "hidden"):
        graph[name].x = torch.tensor([[0.0, 0.0], [0.001, 0.001]])
        graph[name].num_nodes = 2
    for src, dst in (("data", "hidden"), ("hidden", "hidden")):
        edges = graph[src, "to", dst]
        edges.edge_index = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]])
        edges.edge_length = torch.ones(4, 1)
        edges.edge_dirs = torch.zeros(4, 2)
    components = [
        config.model.processor,
        config.model.encoders.inputs.mapper,
        config.model.decoders.observations.mapper,
    ]
    for component in components:
        component.num_channels = 32 if graph_attention_backend == "triton" else 8
        component.graph_attention_backend = graph_attention_backend
    _, dynamic_graph_config = split_graph_config(
        DotDict(config.graph), is_dataset_static={"data": True, "obs": False}, hidden_nodes_name=hidden_nodes_name,
    )
    return model_class(
        model_config=config.model,
        static_graph_data=graph,
        dynamic_graph_config=dynamic_graph_config,
        data_indices=_indices(),
        statistics={"data": {}, "obs": {}},
        n_step_input={"data": 1, "obs": 1},
        n_step_output={"data": 1, "obs": 1},
    )


class TestDynamicTargetEdgeFeatures(unittest.TestCase):
    def test_configuration_validates(self) -> None:
        config = OmegaConf.load(CONFIG_PATH)
        OmegaConf.resolve(config)
        model_config = BaseModelSchema(**config.model)
        graph_config = BaseGraphSchema(**config.graph)
        data_config = DataSchema(**config.data)
        assert model_config.decoders["observations"].target_edge_features == ["level"]
        assert graph_config.edges[0].attributes["level_from_obs"].num_features == 1
        assert data_config.datasets["obs"].forcing == ["level"]
        assert list(data_config.datasets["obs"].processors) == ["normalizer"]

    def test_task_separates_targets_geometry_and_normalized_level(self) -> None:
        raw, templates, forcings = Autoencoder().get_targets(_batch(), data_indices=_indices())
        assert raw["obs"].variables == ["TURBIND", "level"]
        assert templates["obs"].variables == ["TURBIND"]
        assert templates["obs"].flatten().values is None
        assert forcings["obs"].variables == ["level"]
        normalized = InputNormalizer(config={"default": "mean-std"})(forcings["obs"], in_place=False)
        torch.testing.assert_close(normalized.flatten().values[:, 0], torch.tensor([0.0, 1.0, 2.0]))

    def test_forward_edges_and_backward_without_target_leakage(self) -> None:
        backends = [("cpu", "pyg")]
        if torch.cuda.is_available():
            backends.append(("cuda", "pyg"))
            if find_spec("triton") is not None:
                backends.append(("cuda", "triton"))
        for device, backend in backends:
            with self.subTest(device=device, backend=backend):
                torch.manual_seed(42)
                model = _model(backend).to(device)
                task = Autoencoder()
                normalizer = InputNormalizer(config={"default": "mean-std"})
                batch = _batch(device)
                inputs = task.get_inputs(batch, data_indices=_indices())
                _, templates, forcings = task.get_targets(batch, data_indices=_indices())
                forcings = forcings.replace("obs", normalizer(forcings["obs"], in_place=False))
                provider = model.decoder_graph_provider["obs"]
                provider.capture_next_graph("hidden", "obs")
                output = model(inputs, target_forcings=forcings, target_template=templates)
                graph = provider.consume_captured_graph()
                edges = graph["hidden", "to", "obs"]
                expected = forcings["obs"].flatten().values.detach().cpu()[edges.edge_index[1]]
                torch.testing.assert_close(edges.level_from_obs, expected)
                assert [len(sample) for sample in output["obs"].values] == [2, 0, 1]
                assert torch.isfinite(output["obs"].flatten().values).all().item()
                output["obs"].flatten().values.square().sum().backward()
                assert any(p.grad is not None for p in model.parameters())

                # Changing only the observed EDR must not change model inputs or predictions.
                changed = _batch(device, target_value=999.0)
                _, changed_templates, changed_forcings = task.get_targets(changed, data_indices=_indices())
                changed_forcings = changed_forcings.replace(
                    "obs",
                    normalizer(changed_forcings["obs"], in_place=False),
                )
                with torch.no_grad():
                    repeated = model(
                        task.get_inputs(changed, data_indices=_indices()),
                        target_forcings=changed_forcings,
                        target_template=changed_templates,
                    )
                torch.testing.assert_close(output["obs"].flatten().values, repeated["obs"].flatten().values)

    def test_missing_level_fails_clearly(self) -> None:
        model = _model()
        batch = _batch(counts=(2,))
        _, templates, forcings = Autoencoder().get_targets(batch, data_indices=_indices())
        forcings = forcings.replace("obs", forcings["obs"].clone(variables=["other"]))
        with self.assertRaisesRegex(ValueError, "missing target edge forcings"):  # noqa: PT027 - standalone unittest
            model(
                Autoencoder().get_inputs(batch, data_indices=_indices()),
                target_forcings=forcings,
                target_template=templates,
            )

    def test_edge_features_with_per_dataset_hidden_routing(self) -> None:
        model = _model(hidden_nodes_name={"data": "hidden", "obs": "hidden"})
        task = Autoencoder()
        batch = _batch()
        _, templates, forcings = task.get_targets(batch, data_indices=_indices())
        provider = model.decoder_graph_provider["obs"]
        provider.capture_next_graph("hidden", "obs")
        output = model(task.get_inputs(batch, data_indices=_indices()), forcings, templates)
        edges = provider.consume_captured_graph()["hidden", "to", "obs"]
        torch.testing.assert_close(edges.level_from_obs, forcings["obs"].flatten().values[edges.edge_index[1]])
        assert output["obs"].flatten().values.shape == (3, 1)

    def test_ensemble_and_transport_receive_target_edge_forcings(self) -> None:
        variants = [
            (AnemoiEnsModelEncProcDec, "edm_diffusion"),
            (AnemoiTransportModelEncProcDec, "edm_diffusion"),
            (AnemoiTransportModelEncProcDec, "flow_matching"),
        ]
        for model_class, objective in variants:
            with self.subTest(model=model_class.__name__, objective=objective):
                model = _model(model_class=model_class, transport_objective=objective)
                task = Autoencoder()
                batch = _batch()
                inputs = task.get_inputs(batch, data_indices=_indices())
                raw, templates, forcings = task.get_targets(batch, data_indices=_indices())
                normalizer = InputNormalizer(config={"default": "mean-std"})
                forcings = forcings.replace("obs", normalizer(forcings["obs"], in_place=False))
                provider = model.decoder_graph_provider["obs"]
                provider.capture_next_graph("hidden", "obs")
                if isinstance(model, AnemoiTransportModelEncProcDec):
                    noisy_target = Batch({"obs": raw["obs"].select(variables=[0])})
                    output = model(
                        inputs, noisy_target, {"obs": torch.full((batch.batch_size, 1, 1, 1, 1), 0.7)},
                        target_forcings=forcings,
                    )
                else:
                    output = model(inputs, forcings, templates)
                edges = provider.consume_captured_graph()["hidden", "to", "obs"]
                torch.testing.assert_close(edges.level_from_obs, forcings["obs"].flatten().values[edges.edge_index[1]])
                assert torch.isfinite(output["obs"].flatten().values).all().item()
                output["obs"].flatten().values.square().sum().backward()
                assert any(parameter.grad is not None for parameter in model.parameters())

    def test_target_forcing_node_features_preserve_requested_variable_order(self) -> None:
        model = _model()
        target_forcings = _batch()["obs"].clone(variables=["first", "second"])
        with patch("anemoi.models.models.encoder_processor_decoder.gather_tensor") as gather:
            target_forcing_values = model._get_target_forcing_node_features(
                dataset_name="obs",
                target_forcings=target_forcings,
                forcing_names=["second", "first"],
                target_shard_sizes=None,
            )
        torch.testing.assert_close(target_forcing_values, target_forcings.flatten().values[:, [1, 0]])
        gather.assert_not_called()

    def test_target_forcing_node_features_require_requested_forcings(self) -> None:
        model = _model()
        for target_forcings, message in (
            (None, "requires target_forcings"),
            (_batch()["obs"], "missing target edge forcings"),
        ):
            with (
                self.subTest(forcings=target_forcings),
                self.assertRaisesRegex(ValueError, message),  # noqa: PT027 - standalone unittest
            ):
                model._get_target_forcing_node_features(
                    dataset_name="obs",
                    target_forcings=target_forcings,
                    forcing_names=["missing"],
                    target_shard_sizes=None,
                )

    def test_target_forcing_node_features_reject_mismatched_shards(self) -> None:
        model = _model()
        with (
            patch("anemoi.models.models.encoder_processor_decoder.gather_tensor") as gather,
            self.assertRaisesRegex(ValueError, "matching shard sizes"),  # noqa: PT027 - standalone unittest
        ):
            model._get_target_forcing_node_features(
                dataset_name="obs",
                target_forcings=_batch()["obs"],
                forcing_names=["level"],
                target_shard_sizes=[2, 1],
            )
        gather.assert_not_called()

    def test_sharded_edge_values_are_gathered_in_coordinate_order(self) -> None:
        model = _model()
        _, _, forcings = Autoencoder().get_targets(_batch(counts=(2,)), data_indices=_indices())
        source = forcings["obs"].clone(shard_sizes=[[[2, 1]]])
        gathered = torch.tensor([[100.0], [120.0], [140.0]])
        provider = model.decoder_graph_provider["obs"]
        decoder = model.decoder["observations"]
        with (
            patch("anemoi.models.models.encoder_processor_decoder.gather_tensor", return_value=gathered) as gather,
            patch.object(
                provider,
                "get_edges",
                return_value=(torch.zeros(0, 5), torch.empty(2, 0, dtype=torch.long), None),
            ) as edges,
            patch.object(decoder, "forward", return_value=torch.zeros(2, 1)),
        ):
            model._decode_rows(
                "obs",
                torch.zeros(2, 8),
                (torch.zeros(3, 2), torch.zeros(2, 5), [2, 1], (3,), torch.zeros(3)),
                batch_size=1,
                hidden_coordinates=torch.zeros(2, 2),
                hidden_coordinates_batched=torch.zeros(2, 2),
                hidden_batch_sizes=(2,),
                shard_sizes_hidden=None,
                edge_dtype=torch.float32,
                keep_x_dst_sharded=True,
                target_forcings=source,
            )
        torch.testing.assert_close(gather.call_args.args[0], source.flatten().values)
        assert gather.call_args.kwargs["sizes"] == [2, 1]
        torch.testing.assert_close(edges.call_args.kwargs["dst_node_features"], gathered)

    def test_no_level_edge_features_by_default(self) -> None:
        config = OmegaConf.load(CONFIG_PATH)
        del config.model.decoders.observations.target_edge_features
        OmegaConf.resolve(config)
        assert BaseModelSchema(**config.model).decoders["observations"].target_edge_features == []

    def test_globally_empty_observations(self) -> None:
        model = _model()
        batch = _batch(counts=(0,))
        task = Autoencoder()
        _, templates, forcings = task.get_targets(batch, data_indices=_indices())
        output = model(
            task.get_inputs(batch, data_indices=_indices()), target_forcings=forcings, target_template=templates,
        )
        assert output["obs"].flatten().values.shape == (0, 1)


if __name__ == "__main__":
    unittest.main()

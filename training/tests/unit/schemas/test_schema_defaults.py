from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pydantic import BaseModel as PydanticBaseModel

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import build_schema
from anemoi.training.schemas.schema_utils import apply_schema_defaults
from anemoi.training.schemas.schema_utils import prune_undeclared_interpolation_anchors
from anemoi.training.schemas.schema_utils import undeclared_interpolation_anchor_paths


def test_apply_schema_defaults_fills_cutoff_edge_builder_defaults() -> None:
    graph_cfg = OmegaConf.create(
        {
            "overwrite": True,
            "nodes": {
                "data": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.IconNodes",
                        "grid_definition_path": "/tmp/icon_grid.nc",
                    },
                    "attributes": {},
                },
                "hidden": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.HNPNodes",
                        "resolution": 3,
                    },
                    "attributes": {},
                },
            },
            "edges": [
                {
                    "source_name": "data",
                    "target_name": "hidden",
                    "edge_builders": [
                        {
                            "_target_": "anemoi.graphs.edges.CutOffEdges",
                            "cutoff_factor": 0.6,
                            "source_mask_attr_name": None,
                            "target_mask_attr_name": None,
                        },
                    ],
                    "attributes": {},
                },
            ],
            "post_processors": [],
        },
    )

    graph_with_defaults = apply_schema_defaults(graph_cfg, BaseGraphSchema)
    edge_builder = graph_with_defaults.edges[0].edge_builders[0]
    assert edge_builder.max_num_neighbours == 64


def test_apply_schema_defaults_fills_training_model_and_diagnostics_defaults() -> None:
    with initialize(version_base=None, config_path="../../../src/anemoi/training/config", job_name="test_defaults"):
        cfg = compose(config_name="config")
    assert isinstance(cfg, DictConfig)

    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"
        cfg.system.input.graph = "/tmp/graph.pt"
        cfg.training.pop("deterministic", None)
        cfg.training.pop("update_ds_stats_on_ckpt_load", None)
        cfg.model.pop("keep_batch_sharded", None)
        cfg.diagnostics.pop("check_val_every_n_epoch", None)

    cfg_with_defaults = apply_schema_defaults(cfg, BaseSchema)

    assert cfg_with_defaults.training.deterministic is False
    assert cfg_with_defaults.training.update_ds_stats_on_ckpt_load.states is False
    assert cfg_with_defaults.training.update_ds_stats_on_ckpt_load.tendencies is True
    assert cfg_with_defaults.model.keep_batch_sharded is True
    assert cfg_with_defaults.diagnostics.check_val_every_n_epoch == 1


def test_apply_schema_defaults_fills_nested_model_and_callback_options() -> None:
    with initialize(
        version_base=None,
        config_path="../../../src/anemoi/training/config",
        job_name="test_nested_defaults",
    ):
        cfg = compose(config_name="config")
    assert isinstance(cfg, DictConfig)

    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"
        cfg.system.input.graph = "/tmp/graph.pt"
        cfg.system.output.root = "/tmp/out"
        cfg.training.multistep_output = 1

        if "encoder" in cfg.model:
            with open_dict(cfg.model.encoder):
                cfg.model.encoder.pop("gradient_checkpointing", None)
        if "decoder" in cfg.model:
            with open_dict(cfg.model.decoder):
                cfg.model.decoder.pop("gradient_checkpointing", None)

        callbacks = cfg.diagnostics.plot.callbacks
        sample_callback = None
        for callback in callbacks:
            if callback.get("_target_") == "anemoi.training.diagnostics.callbacks.plot.PlotSample":
                sample_callback = callback
                break
        assert sample_callback is not None

        with open_dict(sample_callback):
            sample_callback.pop("focus_area", None)
            sample_callback.pop("every_n_batches", None)
            sample_callback.pop("colormaps", None)

    cfg_with_defaults = apply_schema_defaults(cfg, BaseSchema)

    assert cfg_with_defaults.model.encoder.gradient_checkpointing is True
    assert cfg_with_defaults.model.decoder.gradient_checkpointing is True

    sample_callback = None
    for callback in cfg_with_defaults.diagnostics.plot.callbacks:
        if callback.get("_target_") == "anemoi.training.diagnostics.callbacks.plot.PlotSample":
            sample_callback = callback
            break
    assert sample_callback is not None
    assert sample_callback.focus_area.name is None
    assert sample_callback.focus_area.mask_attr_name is None
    assert sample_callback.focus_area.latlon_bbox is None
    assert sample_callback.every_n_batches is None
    assert sample_callback.colormaps is None


def test_apply_schema_defaults_handles_layer_kernels_and_preserves_triton_settings() -> None:
    with initialize(
        version_base=None,
        config_path="../../../src/anemoi/training/config",
        job_name="test_triton_defaults",
    ):
        cfg = compose(config_name="config", overrides=["model=graphtransformer"])
    assert isinstance(cfg, DictConfig)

    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"
        cfg.system.input.graph = "/tmp/graph.pt"
        cfg.system.output.root = "/tmp/out"
        cfg.training.multistep_output = 1

        # Use a non-default value first, then remove it to verify schema fallback.
        cfg.model.encoder.layer_kernels = {"Attention": {"_target_": "custom.kernel"}}
        cfg.model.decoder.layer_kernels = {"Attention": {"_target_": "custom.kernel"}}
        cfg.model.processor.layer_kernels = {"Attention": {"_target_": "custom.kernel"}}
        cfg.model.encoder.pop("layer_kernels", None)
        cfg.model.decoder.pop("layer_kernels", None)
        cfg.model.processor.pop("layer_kernels", None)

    cfg_with_defaults = apply_schema_defaults(cfg, BaseSchema)

    assert cfg_with_defaults.model.encoder.layer_kernels == {}
    assert cfg_with_defaults.model.decoder.layer_kernels == {}
    assert cfg_with_defaults.model.processor.layer_kernels == {}

    assert cfg_with_defaults.model.encoder.graph_attention_backend == "triton"
    assert cfg_with_defaults.model.decoder.graph_attention_backend == "triton"
    assert cfg_with_defaults.model.processor.graph_attention_backend == "triton"


def test_build_schema_lenient_drops_forwarding_aliases_but_keeps_resolved_values() -> None:
    with initialize(
        version_base=None,
        config_path="../../../src/anemoi/training/config",
        job_name="test_alias_cleanup",
    ):
        cfg = compose(config_name="config", overrides=["model=graphtransformer"])
    assert isinstance(cfg, DictConfig)

    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"
        cfg.system.input.graph = "/tmp/graph.pt"
        cfg.system.output.root = "/tmp/out"
        cfg.training.multistep_output = 1

    parsed = build_schema(cfg)

    assert "layer_kernels" not in parsed.model
    assert "colormaps" not in parsed.diagnostics.plot

    assert isinstance(parsed.model.encoder.layer_kernels, DictConfig | dict)
    assert parsed.model.encoder.layer_kernels == parsed.model.decoder.layer_kernels
    assert parsed.model.encoder.layer_kernels == parsed.model.processor.layer_kernels
    assert "LayerNorm" in parsed.model.encoder.layer_kernels

    sample_callback = None
    for callback in parsed.diagnostics.plot.callbacks:
        if callback.get("_target_") == "anemoi.training.diagnostics.callbacks.plot.PlotSample":
            sample_callback = callback
            break
    assert sample_callback is not None
    assert isinstance(sample_callback.colormaps, DictConfig | dict)


def test_undeclared_interpolation_anchor_paths_minimizes_nested_paths() -> None:
    class RuntimeShape(PydanticBaseModel):
        out_a: int = 1
        out_b: int = 1

    cfg = OmegaConf.create(
        {
            "foo": {"bar": 2},
            "out_a": "${foo}",
            "out_b": "${foo.bar}",
        },
    )

    paths = undeclared_interpolation_anchor_paths(cfg, RuntimeShape)
    assert paths == [("foo",)]

    prune_undeclared_interpolation_anchors(cfg, paths)
    assert "foo" not in cfg


def test_undeclared_interpolation_anchor_paths_keeps_paths_with_declared_descendants() -> None:
    class ItemSchema(PydanticBaseModel):
        value: int = 1

    class RuntimeShape(PydanticBaseModel):
        dynamic: dict[str, ItemSchema]
        out: dict = {}

    cfg = OmegaConf.create(
        {
            "dynamic": {"a": {"value": 3}},
            "out": "${dynamic.a}",
        },
    )

    paths = undeclared_interpolation_anchor_paths(cfg, RuntimeShape)
    assert ("dynamic", "a") not in paths

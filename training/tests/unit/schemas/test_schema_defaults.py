import json

import pytest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import ConfigValidationError
from anemoi.training.schemas.base_schema import apply_runtime_postprocessing
from anemoi.training.schemas.base_schema import build_schema
from anemoi.training.schemas.dataloader import DatasetConfigSchema
from anemoi.training.schemas.schema_utils import apply_schema_defaults
from anemoi.training.schemas.schema_utils import schema_defaults


@pytest.fixture
def base_cfg() -> DictConfig:
    """Minimal valid lenient config from the default Hydra composition."""
    with initialize(version_base=None, config_path="../../../src/anemoi/training/config", job_name="test"):
        cfg = compose(config_name="config")
    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"  # noqa: S108
        cfg.system.input.graph = "/tmp/graph.pt"  # noqa: S108
        cfg.system.output.root = "/tmp/out"  # noqa: S108
        cfg.training.multistep_output = 1
    return cfg


@pytest.fixture
def graphtransformer_cfg() -> DictConfig:
    with initialize(version_base=None, config_path="../../../src/anemoi/training/config", job_name="test_gt"):
        cfg = compose(config_name="config", overrides=["model=graphtransformer"])
    with open_dict(cfg):
        cfg.config_validation = False
        cfg.system.input.dataset = "/tmp/dataset.zarr"  # noqa: S108
        cfg.system.input.graph = "/tmp/graph.pt"  # noqa: S108
        cfg.system.output.root = "/tmp/out"  # noqa: S108
        cfg.training.multistep_output = 1
    return cfg


def _find_callback(callbacks: list, target: str) -> dict | None:
    return next((c for c in callbacks if c.get("_target_") == target), None)


# ---------------------------------------------------------------------------
# DatasetConfigSchema — None dropping in both paths
# ---------------------------------------------------------------------------


def test_dataset_config_schema_drops_none_in_strict_path() -> None:
    """On the strict (pydantic) path, None-valued optional fields must be absent.

    DatasetConfigSchema is forwarded as ``open_dataset(**dataset_config)``.  If
    optional fields appear as ``select=None``, open_dataset treats that as an
    explicit selection containing None rather than "no selection".  The
    ``_exclude_none`` serializer (inherited from NullDropSchema) must strip them.
    """
    schema = DatasetConfigSchema(dataset="/tmp/dataset.zarr")  # noqa: S108
    out = schema.model_dump(by_alias=True)

    assert "dataset" in out
    assert "frequency" not in out, "frequency=None must be dropped, not forwarded as None"
    assert "drop" not in out, "drop=None must be dropped"
    assert "select" not in out, "select=None must be dropped"
    assert "statistics" not in out, "statistics=None must be dropped"


def test_dataset_config_schema_drops_none_in_lenient_path() -> None:
    """On the lenient (schema_defaults) path, None-valued optional fields must be absent.

    ``_skip_null_defaults = True`` on NullDropSchema tells ``schema_defaults`` to
    omit any field whose resolved value is None so the lenient output matches what
    the strict serializer produces.
    """
    out = schema_defaults(DatasetConfigSchema, {"dataset": "/tmp/dataset.zarr"})  # noqa: S108

    assert "frequency" not in out, "frequency default None must not appear in lenient output"
    assert "drop" not in out, "drop default None must not appear in lenient output"
    assert "select" not in out, "select default None must not appear in lenient output"
    assert "statistics" not in out, "statistics default None must not appear in lenient output"


def test_dataset_config_schema_preserves_non_none_values() -> None:
    """Explicitly set non-None values survive both the serializer and schema_defaults."""
    schema = DatasetConfigSchema(dataset="/tmp/dataset.zarr", drop=["tp"], select=["q"])  # noqa: S108
    strict_out = schema.model_dump(by_alias=True)

    assert strict_out["drop"] == ["tp"]
    assert strict_out["select"] == ["q"]

    lenient_out = schema_defaults(
        DatasetConfigSchema,
        {"dataset": "/tmp/dataset.zarr", "drop": ["tp"], "select": ["q"]},  # noqa: S108
    )
    assert lenient_out["drop"] == ["tp"]
    assert lenient_out["select"] == ["q"]


# ---------------------------------------------------------------------------
# Graph schema defaults
# ---------------------------------------------------------------------------


def test_apply_schema_defaults_fills_cutoff_edge_builder_defaults() -> None:
    graph_cfg = OmegaConf.create(
        {
            "overwrite": True,
            "nodes": {
                "data": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.IconNodes",
                        "grid_definition_path": "/tmp/icon_grid.nc",  # noqa: S108
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
    assert graph_with_defaults.edges[0].edge_builders[0].max_num_neighbours == 64


# ---------------------------------------------------------------------------
# Training schema defaults
# ---------------------------------------------------------------------------


def test_apply_schema_defaults_fills_training_defaults(base_cfg: DictConfig) -> None:
    """Missing training fields receive their schema defaults."""
    with open_dict(base_cfg):
        base_cfg.training.pop("deterministic", None)
        base_cfg.training.pop("update_ds_stats_on_ckpt_load", None)
        base_cfg.diagnostics.pop("check_val_every_n_epoch", None)

    cfg = apply_schema_defaults(base_cfg, BaseSchema)

    assert cfg.training.deterministic is False
    assert cfg.training.update_ds_stats_on_ckpt_load.states is False
    assert cfg.training.update_ds_stats_on_ckpt_load.tendencies is True
    assert cfg.diagnostics.check_val_every_n_epoch == 1


def test_apply_schema_defaults_fills_encoder_decoder_defaults(graphtransformer_cfg: DictConfig) -> None:
    """Nested model defaults (encoder/decoder gradient_checkpointing) are injected."""
    with open_dict(graphtransformer_cfg):
        graphtransformer_cfg.model.encoder.pop("gradient_checkpointing", None)
        graphtransformer_cfg.model.decoder.pop("gradient_checkpointing", None)

    cfg = apply_schema_defaults(graphtransformer_cfg, BaseSchema)

    assert cfg.model.encoder.gradient_checkpointing is True
    assert cfg.model.decoder.gradient_checkpointing is True


def test_apply_schema_defaults_callback_missing_fields_get_none(base_cfg: DictConfig) -> None:
    """Optional callback fields omitted by the user default to None, not a nested object."""
    with open_dict(base_cfg):
        sample = _find_callback(
            base_cfg.diagnostics.plot.callbacks,
            "anemoi.training.diagnostics.callbacks.plot.PlotSample",
        )
        assert sample is not None
        sample.pop("focus_area", None)
        sample.pop("every_n_batches", None)
        sample.pop("colormaps", None)

    cfg = apply_schema_defaults(base_cfg, BaseSchema)

    sample = _find_callback(cfg.diagnostics.plot.callbacks, "anemoi.training.diagnostics.callbacks.plot.PlotSample")
    assert sample is not None
    assert sample.focus_area is None
    assert sample.every_n_batches is None
    assert sample.colormaps is None


def test_apply_schema_defaults_preserves_layer_kernels(graphtransformer_cfg: DictConfig) -> None:
    """layer_kernels removed from encoder/decoder/processor receive empty-dict defaults."""
    with open_dict(graphtransformer_cfg):
        graphtransformer_cfg.model.encoder.pop("layer_kernels", None)
        graphtransformer_cfg.model.decoder.pop("layer_kernels", None)
        graphtransformer_cfg.model.processor.pop("layer_kernels", None)

    cfg = apply_schema_defaults(graphtransformer_cfg, BaseSchema)

    assert cfg.model.encoder.layer_kernels == {}
    assert cfg.model.decoder.layer_kernels == {}
    assert cfg.model.processor.layer_kernels == {}
    assert cfg.model.encoder.graph_attention_backend == "triton"


# ---------------------------------------------------------------------------
# Interpolation anchor pruning
# ---------------------------------------------------------------------------


def test_build_schema_lenient_resolves_anchors_and_drops_forwarding_keys(graphtransformer_cfg: DictConfig) -> None:
    """After build_schema in lenient mode, anchor keys not in the schema are removed.

    Their resolved values are present at the reference sites.
    """
    parsed = build_schema(graphtransformer_cfg)

    assert "layer_kernels" not in parsed.model
    assert "colormaps" not in parsed.diagnostics.plot

    assert isinstance(parsed.model.encoder.layer_kernels, DictConfig | dict)
    assert "LayerNorm" in parsed.model.encoder.layer_kernels
    assert parsed.model.encoder.layer_kernels == parsed.model.decoder.layer_kernels

    sample = _find_callback(parsed.diagnostics.plot.callbacks, "anemoi.training.diagnostics.callbacks.plot.PlotSample")
    assert sample is not None
    assert isinstance(sample.colormaps, DictConfig | dict)


# ---------------------------------------------------------------------------
# Lenient mode behaviour
# ---------------------------------------------------------------------------


def test_lenient_accepts_invalid_projection_kind(base_cfg: DictConfig) -> None:
    """Strict mode rejects an invalid projection_kind; lenient mode passes it through."""
    with open_dict(base_cfg):
        base_cfg.diagnostics.plot.projection_kind = "invalid_projection"

    strict = base_cfg.copy()
    with open_dict(strict):
        strict.config_validation = True
    with pytest.raises(ConfigValidationError, match="projection_kind"):
        build_schema(strict)

    lenient = build_schema(base_cfg)
    assert lenient.diagnostics.plot.projection_kind == "invalid_projection"


def test_lenient_drops_extra_fields(base_cfg: DictConfig) -> None:
    """Unknown fields not declared by the schema are pruned in lenient mode."""
    with open_dict(base_cfg.training):
        base_cfg.training.typo_extra_field = "ignored"

    config = build_schema(base_cfg)
    assert not hasattr(config.training, "typo_extra_field")


def test_strict_reports_invalid_discriminator_clearly(base_cfg: DictConfig) -> None:
    """Strict mode surfaces a clear error when the discriminator field is invalid."""
    with open_dict(base_cfg):
        base_cfg.config_validation = True
        base_cfg.training.model_task = "not.a.real.task"

    with pytest.raises(ConfigValidationError, match=r"training\.model_task"):
        build_schema(base_cfg)


# ---------------------------------------------------------------------------
# Lenient == strict equivalence
# ---------------------------------------------------------------------------


def test_lenient_and_strict_produce_same_declared_values(graphtransformer_cfg: DictConfig) -> None:
    """For all fields declared by the schema, lenient and strict produce the same runtime values.

    Both strict and lenient configs are converted to plain Python dicts before
    re-parsing through ``BaseSchema`` so that OmegaConf types do not leak into
    ``Any``-typed or ``extra="allow"`` fields and break JSON serialisation.
    """
    # Build strict from plain dict to avoid OmegaConf leakage into extra/Any fields
    strict_plain = OmegaConf.to_container(graphtransformer_cfg, resolve=True)
    strict_plain["config_validation"] = True
    strict_config = BaseSchema(**strict_plain)
    apply_runtime_postprocessing(strict_config)

    lenient_config = build_schema(graphtransformer_cfg)

    # Convert lenient output to plain Python then re-parse through strict schema
    lenient_plain = OmegaConf.to_container(lenient_config.model_dump(by_alias=True), resolve=True)
    lenient_plain["config_validation"] = True
    strict_from_lenient = BaseSchema(**lenient_plain)

    # Compare parsed JSON (not raw strings) to be insensitive to key ordering
    assert json.loads(strict_config.model_dump_json(by_alias=True)) == json.loads(
        strict_from_lenient.model_dump_json(by_alias=True),
    )

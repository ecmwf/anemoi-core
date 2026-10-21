# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Removed config keys raise an error that points to ``training.checkpoint``."""

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.schemas.base_schema import _DEPRECATED_KEYS
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.training.schemas.training import CheckpointPipelineSchema

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOCAL_SOURCE = "anemoi.training.checkpoint.sources.local.LocalSource"
_LOADERS = "anemoi.training.checkpoint.loading.strategies"
_FREEZING = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"

# The training.checkpoint block that the hint for each removed key tells the user to write.
_REPLACEMENTS = {
    "training.run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": False}},
    "training.fork_run_id": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc", "fork": True}},
    "system.input.warm_start": {"source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"}},
    "training.load_weights_only": {
        "source": {"_target_": _RUN_SOURCE, "run_id": "abc"},
        "loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"},
    },
    "training.transfer_learning": {
        "source": {"_target_": _LOCAL_SOURCE, "path": "/scratch/last.ckpt"},
        "loading": {"_target_": f"{_LOADERS}.TransferLearningLoader", "skip_mismatched": True},
    },
    "training.submodules_to_freeze": {
        "modifiers": [{"_target_": _FREEZING, "submodule_root": "model.model", "submodules_to_freeze": ["encoder"]}],
    },
}


def _nested(path_parts: tuple[str, ...], value: object) -> dict:
    """Build a nested dict from path parts."""
    config: dict = {}
    node = config
    for part in path_parts[:-1]:
        node = node.setdefault(part, {})
    node[path_parts[-1]] = value
    return config


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
@pytest.mark.parametrize(
    ("path_parts", "hint_fragment"),
    [
        (("training", "run_id"), "training.run_id has been removed"),
        (("training", "fork_run_id"), "training.fork_run_id has been removed"),
        (("system", "input", "warm_start"), "system.input.warm_start has been removed"),
        (("training", "load_weights_only"), "training.load_weights_only has been removed"),
        (("training", "transfer_learning"), "training.transfer_learning has been removed"),
        (("training", "submodules_to_freeze"), "training.submodules_to_freeze has been removed"),
    ],
)
def test_removed_key_raises(
    schema_cls: type,
    path_parts: tuple[str, ...],
    hint_fragment: str,
) -> None:
    """A removed key raises with a hint."""
    config = _nested(path_parts, "some-value")
    with pytest.raises(ValidationError, match=hint_fragment):
        schema_cls(**config)


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
@pytest.mark.parametrize(
    "path_parts",
    [
        ("training", "run_id"),
        ("training", "fork_run_id"),
        ("system", "input", "warm_start"),
        ("training", "load_weights_only"),
        ("training", "transfer_learning"),
        ("training", "submodules_to_freeze"),
    ],
)
def test_removed_key_present_as_null_still_raises(schema_cls: type, path_parts: tuple[str, ...]) -> None:
    """A removed key set to ``null`` also raises."""
    config = _nested(path_parts, None)
    with pytest.raises(ValidationError, match="has been removed"):
        schema_cls(**config)


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
def test_submodules_to_freeze_hint_names_the_submodule_root(schema_cls: type) -> None:
    """The ``submodules_to_freeze`` hint names ``submodule_root``.

    Without the root, the paths do not resolve and nothing is frozen.
    """
    config = _nested(("training", "submodules_to_freeze"), ["encoder"])
    with pytest.raises(ValidationError, match=r"submodule_root: model\.model") as excinfo:
        schema_cls(**config)
    assert "FreezingModifierStage" in str(excinfo.value)


def test_every_removed_key_has_a_replacement() -> None:
    assert set(_REPLACEMENTS) == set(_DEPRECATED_KEYS)


@pytest.mark.parametrize(
    ("removed_key", "expected_stages"),
    [
        ("training.run_id", ["ResolveOnlySource"]),
        ("training.fork_run_id", ["ResolveOnlySource"]),
        ("system.input.warm_start", ["ResolveOnlySource"]),
        ("training.load_weights_only", ["RunIdSource", "WeightsOnlyLoader"]),
        ("training.transfer_learning", ["LocalSource", "TransferLearningLoader"]),
        ("training.submodules_to_freeze", ["FreezingModifierStage"]),
    ],
)
def test_replacement_config_validates_and_builds(removed_key: str, expected_stages: list[str]) -> None:
    """The config that the hint for each removed key gives is valid and builds a pipeline."""
    checkpoint_block = _REPLACEMENTS[removed_key]

    CheckpointPipelineSchema(**checkpoint_block)
    pipeline = build_checkpoint_pipeline(OmegaConf.create({"training": {"checkpoint": checkpoint_block}}))

    assert [type(stage).__name__ for stage in pipeline.stages] == expected_stages

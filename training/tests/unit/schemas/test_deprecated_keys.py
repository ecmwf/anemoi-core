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

_RUN_SOURCE = {"_target_": "anemoi.training.checkpoint.sources.run.RunIdSource", "run_id": "abc"}
_LOCAL_SOURCE = {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": "/scratch/last.ckpt"}
_LOADERS = "anemoi.training.checkpoint.loading.strategies"

# The ``training.checkpoint`` block that each removal hint gives, and the stages it builds.
# A resume builds no stages: Trainer.fit(ckpt_path=) loads the checkpoint.
_REPLACEMENTS = {
    "training.run_id": ({"source": {**_RUN_SOURCE, "fork": False}}, []),
    "training.fork_run_id": ({"source": {**_RUN_SOURCE, "fork": True}}, []),
    "system.input.warm_start": ({"source": _LOCAL_SOURCE}, []),
    "training.load_weights_only": (
        {"source": _RUN_SOURCE, "loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"}},
        ["RunIdSource", "WeightsOnlyLoader"],
    ),
    "training.transfer_learning": (
        {
            "source": _LOCAL_SOURCE,
            "loading": {"_target_": f"{_LOADERS}.TransferLearningLoader", "skip_mismatched": True},
        },
        ["LocalSource", "TransferLearningLoader"],
    ),
    "training.submodules_to_freeze": (
        {
            "modifiers": [
                {
                    "_target_": "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage",
                    "submodules_to_freeze": ["model.model.encoder"],
                },
            ],
        },
        ["FreezingModifierStage"],
    ),
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
def test_submodules_to_freeze_hint_names_the_full_path(schema_cls: type) -> None:
    """The ``submodules_to_freeze`` hint gives paths from the LightningModule."""
    config = _nested(("training", "submodules_to_freeze"), ["encoder"])
    with pytest.raises(ValidationError, match=r"model\.model\.<submodule>") as excinfo:
        schema_cls(**config)
    assert "FreezingModifierStage" in str(excinfo.value)


def test_every_removed_key_has_a_replacement() -> None:
    assert set(_REPLACEMENTS) == set(_DEPRECATED_KEYS)


@pytest.mark.parametrize(("checkpoint_block", "expected_stages"), _REPLACEMENTS.values(), ids=_REPLACEMENTS.keys())
def test_replacement_config_validates_and_builds(checkpoint_block: dict, expected_stages: list[str]) -> None:
    """The replacement that a removal hint gives is a valid pipeline config."""
    CheckpointPipelineSchema(**checkpoint_block)

    pipeline = build_checkpoint_pipeline(OmegaConf.create({"training": {"checkpoint": checkpoint_block}}))

    assert [type(stage).__name__ for stage in pipeline.stages] == expected_stages

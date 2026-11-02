# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Tests to make sure checkpoint load lineage storage works as expected."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anemoi.training.checkpoint.lineage import record_checkpoint_lineage
from anemoi.training.checkpoint.runner import run_checkpoint_pipeline
from anemoi.training.diagnostics.callbacks.provenance import ParentUUIDCallback

if TYPE_CHECKING:
    from pathlib import Path

_LOADERS = "anemoi.training.checkpoint.loading.strategies"


class _TrainingModule(torch.nn.Module):
    """Training module with one layer, so a loader has parameters to restore."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.hparams = {"metadata": {"checkpoint_lineage": []}}


def test_a_load_is_appended_to_the_lineage() -> None:
    loaded_uuid = "8ec990b8-233d-4a0e-990d-35e603022517"
    earlier_uuid = "426ae8d1-7ade-4ec0-98eb-27ef7fda22c6"
    cold_start = {"loading_strategy": "cold_start", "parent_uuid": loaded_uuid}

    module = _TrainingModule()
    record_checkpoint_lineage(
        module=module,
        checkpoint={"hyper_parameters": {"metadata": {"uuid": loaded_uuid}}},
        loading_strategy="cold_start",
    )
    assert module.hparams["metadata"]["checkpoint_lineage"] == [cold_start]
    assert module.hparams["metadata"]["parent_uuid"] == loaded_uuid

    # Checkpoints saved before lineage existed have no uuid.
    module = _TrainingModule()
    record_checkpoint_lineage(
        module=module,
        checkpoint={"hyper_parameters": {"metadata": {}}},
        loading_strategy="cold_start",
    )
    assert module.hparams["metadata"]["parent_uuid"] is None

    # History on the loaded file is copied, then this new load is added.
    inherited_lineage = [{"loading_strategy": "warm_start", "parent_uuid": earlier_uuid}]
    original = [dict(entry) for entry in inherited_lineage]
    module = _TrainingModule()
    record_checkpoint_lineage(
        module=module,
        checkpoint={"hyper_parameters": {"metadata": {"uuid": loaded_uuid, "checkpoint_lineage": inherited_lineage}}},
        loading_strategy="cold_start",
    )
    assert inherited_lineage == original
    assert module.hparams["metadata"]["checkpoint_lineage"] == [*original, cold_start]


def test_a_lightning_resume_is_recorded_as_a_warm_start() -> None:
    loaded_uuid = "4a0b6254-80b7-4ff2-bfbc-32f8f2a63b1f"
    module = _TrainingModule()
    ParentUUIDCallback().on_load_checkpoint(None, module, {"hyper_parameters": {"metadata": {"uuid": loaded_uuid}}})
    assert module.hparams["metadata"]["checkpoint_lineage"] == [
        {"loading_strategy": "warm_start", "parent_uuid": loaded_uuid},
    ]


def test_the_pipeline_records_only_the_load_it_performs(tmp_path: Path) -> None:
    parent_uuid = "6ce6fec3-96df-46e2-9ff5-735f6a3f1b32"

    def run(loader: str) -> tuple[_TrainingModule, object]:
        model = _TrainingModule()
        checkpoint = tmp_path / f"{loader}.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": {"metadata": {"uuid": parent_uuid}},
            },
            checkpoint,
        )
        config = {
            "training": {
                "checkpoint": {
                    "source": {
                        "_target_": "anemoi.training.checkpoint.sources.local.LocalSource",
                        "path": str(checkpoint),
                    },
                    "loading": {"_target_": f"{_LOADERS}.{loader}"},
                },
            },
        }
        return model, run_checkpoint_pipeline(model, config)

    cold, cold_result = run("ColdStartLoader")
    assert cold.hparams["metadata"]["checkpoint_lineage"] == [
        {"loading_strategy": "cold_start", "parent_uuid": parent_uuid},
    ]
    assert cold_result.ckpt_path is None

    warm, warm_result = run("WarmStartLoader")
    assert warm.hparams["metadata"]["checkpoint_lineage"] == []
    assert warm_result.ckpt_path is not None

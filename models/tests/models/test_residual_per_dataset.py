# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from anemoi.models.models.base import BaseGraphModel
from anemoi.models.schemas.models import BaseModelSchema

SKIP = {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}
NOSKIP = {"_target_": "anemoi.models.layers.residual.NoSkipConnection"}


def _resolve(residual_config: dict, dataset_names: list[str]) -> dict:
    model = SimpleNamespace(dataset_names=dataset_names)
    return BaseGraphModel._residual_configs_per_dataset(model, residual_config)


def test_shared_residual_config_applies_to_every_dataset() -> None:
    assert _resolve(SKIP, ["data", "swiss"]) == {"data": SKIP, "swiss": SKIP}


def test_per_dataset_residual_config() -> None:
    assert _resolve({"data": SKIP, "swiss": NOSKIP}, ["data", "swiss"]) == {"data": SKIP, "swiss": NOSKIP}


@pytest.mark.parametrize(
    "residual_config",
    [{"data": SKIP}, {"data": SKIP, "swiss": NOSKIP, "other": SKIP}],
    ids=["missing", "unknown"],
)
def test_per_dataset_residual_config_must_match_datasets(residual_config: dict) -> None:
    with pytest.raises(ValueError, match="one entry per dataset"):
        _resolve(residual_config, ["data", "swiss"])


def test_residual_schema_accepts_shared_and_per_dataset() -> None:
    adapter = TypeAdapter(BaseModelSchema.model_fields["residual"].annotation)
    assert type(adapter.validate_python(SKIP)).__name__ == "SkipConnectionSchema"
    per_dataset = adapter.validate_python({"data": SKIP, "swiss": NOSKIP})
    assert {k: type(v).__name__ for k, v in per_dataset.items()} == {
        "data": "SkipConnectionSchema",
        "swiss": "NoSkipConnectionSchema",
    }

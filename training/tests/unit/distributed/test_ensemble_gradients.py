# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare ensemble-sharded DDP gradients with an unsharded reference on CPU."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from anemoi.training.distributed.strategy import register_gradient_scaling_hooks
from anemoi.training.testing import _EnsembleModel
from anemoi.training.testing import _spawn_ensemble_ddp_gradients_match_unsharded_reference


def test_gradient_scaling_hook_exclusions_are_explicit() -> None:
    """None scales every parameter, while model-sharding exclusions are opt-in."""
    scale_all_model = _EnsembleModel()
    register_gradient_scaling_hooks(scale_all_model, 3)
    sum(parameter.sum() for parameter in scale_all_model.parameters()).backward()

    for parameter in scale_all_model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 3))

    model_sharding_model = _EnsembleModel()
    register_gradient_scaling_hooks(
        model_sharding_model,
        3,
        skip_grad_scaling=("trainable", "no_gradscaling"),
    )
    sum(parameter.sum() for parameter in model_sharding_model.parameters()).backward()

    torch.testing.assert_close(model_sharding_model.weight.grad, torch.full_like(model_sharding_model.weight, 3))
    torch.testing.assert_close(
        model_sharding_model.trainable.grad,
        torch.ones_like(model_sharding_model.trainable),
    )
    torch.testing.assert_close(
        model_sharding_model.no_gradscaling.grad,
        torch.ones_like(model_sharding_model.no_gradscaling),
    )


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
def test_ensemble_ddp_gradients_match_unsharded_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DDP must average data groups while summing model and ensemble contributions."""
    # Limit thread pools before the spawned workers import numerical libraries.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _spawn_ensemble_ddp_gradients_match_unsharded_reference(tmp_path)

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.training.optimizers.AdEMAMix import AdEMAMix
from anemoi.training.train.tasks.base import BaseGraphModule


def _mocked_module(optimizer_cfg: dict | None = None) -> BaseGraphModule:
    module = MagicMock(spec=BaseGraphModule)
    module.lr = 0.001
    module.lr_min = 1e-5
    module.lr_iterations = 1000
    module.lr_warmup = 100
    module.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    module.config = OmegaConf.create(
        {
            "training": {
                "optimizer": optimizer_cfg
                or {
                    "_target_": "torch.optim.AdamW",
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            },
        },
    )
    module._create_optimizer_from_config = BaseGraphModule._create_optimizer_from_config.__get__(module)
    module._optimizer_uses_external_scheduler = BaseGraphModule._optimizer_uses_external_scheduler.__get__(module)
    module._create_scheduler = BaseGraphModule._create_scheduler.__get__(module)
    module.configure_optimizers = BaseGraphModule.configure_optimizers.__get__(module)
    return module


def test_ademamix_creator_parameters_instantiate() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdEMAMix(
        [param],
        lr=1e-3,
        betas=(0.9, 0.95, 0.9999),
        alpha=8.0,
        beta3_warmup=10_000,
        alpha_warmup=10_000,
        eps=1e-8,
        weight_decay=0.01,
    )

    assert optimizer.defaults["betas"] == (0.9, 0.95, 0.9999)
    assert optimizer.defaults["alpha"] == pytest.approx(8.0)
    assert optimizer.defaults["beta3_warmup"] == 10_000
    assert optimizer.defaults["alpha_warmup"] == 10_000


def test_soap_adapter_runs_two_steps() -> None:
    from anemoi.training.optimizers.soap import SOAP

    model = torch.nn.Linear(3, 1)
    optimizer = SOAP(
        model.parameters(),
        lr=3e-3,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
    )

    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.ones(4, 3)).square().mean()
        loss.backward()
        optimizer.step()

    assert isinstance(optimizer, torch.optim.Optimizer)


def test_schedule_free_adamw_skips_cosine_scheduler_and_supports_modes() -> None:
    schedule_free_cfg = {
        "_target_": "schedulefree.AdamWScheduleFree",
        "lr": 0.0025,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "r": 0.0,
        "weight_lr_power": 2.0,
    }
    module = _mocked_module(schedule_free_cfg)

    optimizers, schedulers = module.configure_optimizers()

    optimizer = optimizers[0]
    assert schedulers == []
    assert optimizer.__class__.__name__ == "AdamWScheduleFree"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)
    assert callable(getattr(optimizer, "train"))
    assert callable(getattr(optimizer, "eval"))


def test_distributed_shampoo_support_reports_torch_compatibility() -> None:
    from anemoi.training.optimizers.optimizer_availability import check_distributed_shampoo_support

    support = check_distributed_shampoo_support()

    assert support.optimizer == "distributed_shampoo"
    if not support.available:
        assert support.reason
        assert "torch" in support.reason.lower() or "import" in support.reason.lower()

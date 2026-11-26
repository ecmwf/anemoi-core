# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import OmegaConf
from pytest_mock import MockerFixture
from pytorch_lightning.utilities.types import LRSchedulerConfig
from timm.scheduler import CosineLRScheduler

from anemoi.training.optimizers.AdEMAMix import AdEMAMix
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.base import resolve_learning_rates
from anemoi.training.train.methods.ensemble import EnsembleTraining

_ADAM_CFG = OmegaConf.create({"_target_": "torch.optim.Adam", "betas": [0.9, 0.95], "weight_decay": 0.1})


@pytest.fixture
def mocked_module(mocker: MockerFixture) -> BaseTrainingModule:
    """Create a lightweight mock BaseTrainingModule instance with real methods bound."""
    module = mocker.MagicMock(spec=BaseTrainingModule)

    module.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    module.learning_rate = 0.001
    module.learning_rate_minimum = None

    # Default config: no scheduler
    module.config = mocker.MagicMock()
    module.config.training.optimization.lr_scheduler = None

    # Bind real methods from the class so they work on this mock
    module.configure_optimizers = BaseTrainingModule.configure_optimizers.__get__(module)
    module.log_optimizer = BaseTrainingModule.log_optimizer
    module.lr_scheduler_step = BaseTrainingModule.lr_scheduler_step.__get__(module)
    module.current_epoch = 0
    module.trainer = mocker.MagicMock()
    module.trainer.global_step = 0

    return module


# ---- Tests ----


def test_create_optimizer_from_config(mocked_module: BaseTrainingModule) -> None:
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG

    result = mocked_module.configure_optimizers()

    assert isinstance(result, torch.optim.Adam)
    param_group = result.param_groups[0]
    assert param_group["lr"] == pytest.approx(mocked_module.learning_rate)
    assert param_group["weight_decay"] == pytest.approx(0.1)
    assert result.defaults["betas"] == (0.9, 0.95)


def test_create_optimizer_from_config_ademamix(mocked_module: BaseTrainingModule) -> None:
    mocked_module.config.training.optimization.optimizer = OmegaConf.create(
        {
            "_target_": "anemoi.training.optimizers.AdEMAMix.AdEMAMix",
            "betas": [0.9, 0.95, 0.9999],
            "weight_decay": 0.1,
        },
    )

    result = mocked_module.configure_optimizers()

    assert isinstance(result, torch.optim.Optimizer)
    param_group = result.param_groups[0]
    assert param_group["lr"] == pytest.approx(mocked_module.learning_rate)
    assert param_group["weight_decay"] == pytest.approx(0.1)
    assert result.defaults["betas"] == (0.9, 0.95, 0.9999)


def test_create_optimizer_from_config_invalid(mocked_module: BaseTrainingModule) -> None:
    mocked_module.config.training.optimization.optimizer = OmegaConf.create(
        {"_target_": "nonexistent.OptimizerClass"},
    )
    with pytest.raises(Exception, match="Error locating target"):
        mocked_module.configure_optimizers()


def test_create_scheduler(mocked_module: BaseTrainingModule) -> None:
    """Ensure cosine scheduler is constructed correctly via configure_optimizers."""
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG
    mocked_module.config.training.optimization.lr_scheduler = OmegaConf.create(
        {
            "_target_": "timm.scheduler.CosineLRScheduler",
            "lr_min": 1e-5,
            "t_initial": 1000,
            "warmup_t": 100,
        },
    )
    mocked_module.config.training.optimization.pl_lr_scheduler = OmegaConf.create({"interval": "epoch"})

    optimizers, schedulers = mocked_module.configure_optimizers()
    optimizer = optimizers[0]
    scheduler_dict = schedulers[0]

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler_dict["scheduler"], CosineLRScheduler)
    assert scheduler_dict["scheduler"].optimizer is optimizer
    assert scheduler_dict["interval"] == "epoch"


def test_create_timm_scheduler_defaults_to_step_interval(
    mocked_module: BaseTrainingModule,
) -> None:
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG
    mocked_module.config.training.optimization.lr_scheduler = OmegaConf.create(
        {
            "_target_": "timm.scheduler.CosineLRScheduler",
            "lr_min": 1e-5,
            "t_initial": 1000,
            "warmup_t": 100,
        },
    )
    mocked_module.config.training.optimization.pl_lr_scheduler = OmegaConf.create({"interval": "step"})

    optimizers, schedulers = mocked_module.configure_optimizers()
    optimizer = optimizers[0]
    scheduler_dict = schedulers[0]

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler_dict["scheduler"], CosineLRScheduler)
    assert scheduler_dict["scheduler"].optimizer is optimizer
    assert scheduler_dict["interval"] == "step"


def test_scheduler_gets_minimum_learning_rate(mocked_module: BaseTrainingModule) -> None:
    mocked_module.learning_rate_minimum = 2e-6
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG
    mocked_module.config.training.optimization.lr_scheduler = OmegaConf.create(
        {"_target_": "timm.scheduler.CosineLRScheduler", "t_initial": 1000, "warmup_t": 100},
    )
    mocked_module.config.training.optimization.pl_lr_scheduler = OmegaConf.create({"interval": "step"})

    _, schedulers = mocked_module.configure_optimizers()

    assert schedulers[0]["scheduler"].lr_min == pytest.approx(2e-6)


def test_scheduler_rejects_minimum_set_twice(mocked_module: BaseTrainingModule) -> None:
    mocked_module.learning_rate_minimum = 2e-6
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG
    mocked_module.config.training.optimization.lr_scheduler = OmegaConf.create(
        {"_target_": "timm.scheduler.CosineLRScheduler", "lr_min": 1e-5, "t_initial": 1000},
    )

    with pytest.raises(ValueError, match="lr_min"):
        mocked_module.configure_optimizers()


def _optimization_config(**learning_rates: float | None) -> OmegaConf:
    keys = ("learning_rate", "learning_rate_per_replica", "learning_rate_minimum", "learning_rate_per_replica_minimum")
    return OmegaConf.create({key: learning_rates.get(key) for key in keys})


@pytest.mark.parametrize(
    ("learning_rates", "expected"),
    [
        ({"learning_rate": 1e-3, "learning_rate_minimum": 1e-6}, (1e-3, 1e-6)),
        ({"learning_rate_per_replica": 1e-4, "learning_rate_per_replica_minimum": 1e-7}, (4e-4, 4e-7)),
        ({"learning_rate": 1e-3, "learning_rate_per_replica_minimum": 1e-7}, (1e-3, 4e-7)),
        ({"learning_rate_per_replica": 1e-4, "learning_rate_minimum": 1e-6}, (4e-4, 1e-6)),
        ({"learning_rate": 1e-3}, (1e-3, None)),
    ],
)
def test_resolve_learning_rates(learning_rates: dict, expected: tuple[float, float | None]) -> None:
    peak, minimum = resolve_learning_rates(_optimization_config(**learning_rates), num_model_replicas=4)

    assert peak == pytest.approx(expected[0])
    assert minimum == (None if expected[1] is None else pytest.approx(expected[1]))


@pytest.mark.parametrize(
    "learning_rates",
    [
        {"learning_rate": 1e-3, "learning_rate_per_replica": 1e-4},
        {"learning_rate": 1e-3, "learning_rate_minimum": 1e-6, "learning_rate_per_replica_minimum": 1e-7},
    ],
)
def test_resolve_learning_rates_rejects_both_keys_of_a_pair(learning_rates: dict) -> None:
    with pytest.raises(ValueError, match="Set one of them and the other to null"):
        resolve_learning_rates(_optimization_config(**learning_rates), num_model_replicas=4)


def test_resolve_learning_rates_requires_peak() -> None:
    with pytest.raises(ValueError, match="learning_rate_per_replica"):
        resolve_learning_rates(_optimization_config(learning_rate_minimum=1e-6), num_model_replicas=4)


@pytest.mark.parametrize(
    ("module_class", "expected"),
    [(BaseTrainingModule, 8), (EnsembleTraining, 4)],
)
def test_num_model_replicas(mocker: MockerFixture, module_class: type, expected: int) -> None:
    module = mocker.MagicMock(spec=module_class)
    module.config = OmegaConf.create(
        {
            "system": {
                "hardware": {
                    "num_nodes": 2,
                    "num_gpus_per_node": 8,
                    "num_gpus_per_model": 2,
                    "num_gpus_per_ensemble": 4,
                },
            },
        },
    )

    assert module_class._num_model_replicas(module) == expected


def test_lr_scheduler_step_uses_step_update_for_step_interval(
    mocked_module: BaseTrainingModule,
    mocker: MockerFixture,
) -> None:
    optimizer = torch.optim.Adam(
        mocked_module.parameters(),
        lr=mocked_module.learning_rate,
    )
    scheduler = CosineLRScheduler(optimizer, t_initial=1000, warmup_t=100, t_in_epochs=False)
    step_update = mocker.patch.object(scheduler, "step_update")
    step = mocker.patch.object(scheduler, "step")
    mocked_module.trainer.global_step = 123
    mocked_module.trainer.lr_scheduler_configs = [LRSchedulerConfig(scheduler=scheduler, interval="step")]

    mocked_module.lr_scheduler_step(scheduler)

    step_update.assert_called_once_with(123, None)
    step.assert_not_called()


def test_lr_scheduler_step_uses_epoch_step_for_epoch_interval(
    mocked_module: BaseTrainingModule,
    mocker: MockerFixture,
) -> None:
    optimizer = torch.optim.Adam(
        mocked_module.parameters(),
        lr=mocked_module.learning_rate,
    )
    scheduler = CosineLRScheduler(optimizer, t_initial=10, warmup_t=1, t_in_epochs=True)
    step_update = mocker.patch.object(scheduler, "step_update")
    step = mocker.patch.object(scheduler, "step")
    mocked_module.current_epoch = 4
    mocked_module.trainer.lr_scheduler_configs = [LRSchedulerConfig(scheduler=scheduler, interval="epoch")]

    mocked_module.lr_scheduler_step(scheduler)

    step.assert_called_once_with(5, None)
    step_update.assert_not_called()


def test_ademamix_single_step_numerical() -> None:
    # --- Setup a single scalar parameter ---
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = AdEMAMix([param], lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0)

    # --- Define a simple loss ---
    loss = (param - 5) ** 2 / 2  # grad = param - 5 = -4
    loss.backward()

    # --- Capture gradient manually ---
    grad = param.grad.clone().detach()

    # --- Compute expected update manually (step = 1) ---
    beta1, beta2, beta3 = (0.9, 0.999, 0.9999)
    eps = 1e-8
    alpha = 2.0
    lr = 1e-3

    # Initialize states
    exp_avg_fast = (1 - beta1) * grad
    exp_avg_slow = (1 - beta3) * grad
    exp_avg_sq = (1 - beta2) * (grad * grad)

    bias_correction1 = 1 - beta1
    bias_correction2 = 1 - beta2

    denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2))) + eps
    update = (exp_avg_fast / bias_correction1 + alpha * exp_avg_slow) / denom

    expected_param = 1.0 - lr * update.item()

    # --- Step optimizer ---
    optimizer.step()

    # --- Compare ---
    assert torch.allclose(param, torch.tensor([expected_param]), atol=1e-8)

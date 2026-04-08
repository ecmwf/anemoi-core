import pytest
import torch
from omegaconf import OmegaConf
from pytest_mock import MockerFixture
from timm.scheduler import CosineLRScheduler

from anemoi.training.optimizers.AdEMAMix import AdEMAMix
from anemoi.training.train.tasks.base import BaseGraphModule

_ADAM_CFG = OmegaConf.create({"_target_": "torch.optim.Adam", "betas": [0.9, 0.95], "weight_decay": 0.1})


@pytest.fixture
def mocked_module(mocker: MockerFixture) -> BaseGraphModule:
    """Create a lightweight mock BaseGraphModule instance with real methods bound."""
    module = mocker.MagicMock(spec=BaseGraphModule)

    module.effective_lr = 0.001
    module.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

    # Default config: no scheduler
    module.config = mocker.MagicMock()
    module.config.training.optimization.lr_scheduler = None

    # Bind real methods from the class so they work on this mock
    module.configure_optimizers = BaseGraphModule.configure_optimizers.__get__(module)
    module.log_optimizer = BaseGraphModule.log_optimizer
    module.lr_scheduler_step = BaseGraphModule.lr_scheduler_step.__get__(module)
    module.current_epoch = 0
    module._lr_scheduler_interval = "epoch"
    module.trainer = mocker.MagicMock()
    module.trainer.global_step = 0

    return module


# ---- Tests ----


def test_create_optimizer_from_config(mocked_module: BaseGraphModule) -> None:
    mocked_module.config.training.optimization.optimizer = _ADAM_CFG

    result = mocked_module.configure_optimizers()

    assert isinstance(result, torch.optim.Adam)
    param_group = result.param_groups[0]
    assert param_group["lr"] == pytest.approx(mocked_module.effective_lr)
    assert param_group["weight_decay"] == pytest.approx(0.1)
    assert result.defaults["betas"] == (0.9, 0.95)


def test_create_optimizer_from_config_ademamix(mocked_module: BaseGraphModule) -> None:
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
    assert param_group["lr"] == pytest.approx(mocked_module.effective_lr)
    assert param_group["weight_decay"] == pytest.approx(0.1)
    assert result.defaults["betas"] == (0.9, 0.95, 0.9999)


def test_create_optimizer_from_config_invalid(mocked_module: BaseGraphModule) -> None:
    mocked_module.config.training.optimization.optimizer = OmegaConf.create(
        {"_target_": "nonexistent.OptimizerClass"},
    )
    with pytest.raises(Exception, match="Error locating target"):
        mocked_module.configure_optimizers()


def test_create_scheduler(mocked_module: BaseGraphModule) -> None:
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
    mocked_module: BaseGraphModule,
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
    mocked_module.config.training.optimization.pl_lr_scheduler = None

    optimizers, schedulers = mocked_module.configure_optimizers()
    optimizer = optimizers[0]
    scheduler_dict = schedulers[0]

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler_dict.scheduler, CosineLRScheduler)
    assert scheduler_dict.scheduler.optimizer is optimizer
    assert scheduler_dict.interval == "epoch"
    assert mocked_module._lr_scheduler_interval == "step"


def test_lr_scheduler_step_uses_step_update_for_step_interval(
    mocked_module: BaseGraphModule,
    mocker: MockerFixture,
) -> None:
    optimizer = torch.optim.Adam(mocked_module.parameters(), lr=mocked_module.effective_lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=1000, warmup_t=100, t_in_epochs=False)
    step_update = mocker.patch.object(scheduler, "step_update")
    step = mocker.patch.object(scheduler, "step")
    mocked_module.trainer.global_step = 123
    mocked_module._lr_scheduler_interval = "step"

    mocked_module.lr_scheduler_step(scheduler)

    step_update.assert_called_once_with(123, None)
    step.assert_not_called()


def test_lr_scheduler_step_uses_epoch_step_for_epoch_interval(
    mocked_module: BaseGraphModule,
    mocker: MockerFixture,
) -> None:
    optimizer = torch.optim.Adam(mocked_module.parameters(), lr=mocked_module.effective_lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=10, warmup_t=1, t_in_epochs=True)
    step_update = mocker.patch.object(scheduler, "step_update")
    step = mocker.patch.object(scheduler, "step")
    mocked_module.current_epoch = 4
    mocked_module._lr_scheduler_interval = "epoch"

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

import pytest
import torch
from omegaconf import OmegaConf
from pytest_mock import MockerFixture
from timm.scheduler import CosineLRScheduler

from anemoi.training.optimizers.AdEMAMix import AdEMAMix
from anemoi.training.train.tasks.base import BaseGraphModule


@pytest.fixture
def mocked_module(mocker: MockerFixture) -> BaseGraphModule:
    """Create a lightweight mock BaseGraphModule instance with real methods bound."""
    module = mocker.MagicMock(spec=BaseGraphModule)

    # Inject minimal attributes required by the tested functions
    module.lr = 0.001
    module.lr_min = 1e-5
    module.lr_iterations = 1000
    module.lr_warmup = 100
    module.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

    # Bind real methods from the class so they work on this mock
    module._create_optimizer_from_config = BaseGraphModule._create_optimizer_from_config.__get__(module)
    module._create_scheduler = BaseGraphModule._create_scheduler.__get__(module)
    module.configure_optimizers = BaseGraphModule.configure_optimizers.__get__(module)

    return module


# ---- Tests ----


def test_create_optimizer_from_config(mocked_module: BaseGraphModule) -> None:

    optimizer_cfg = OmegaConf.create(
        {
            "_target_": "torch.optim.Adam",
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
        },
    )

    expected_wd = optimizer_cfg.weight_decay
    expected_betas = tuple(optimizer_cfg.betas)

    cfg_copy = OmegaConf.create(OmegaConf.to_container(optimizer_cfg, resolve=True))

    optimizer = mocked_module._create_optimizer_from_config(cfg_copy)

    param_group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.Adam)
    assert param_group["lr"] == pytest.approx(mocked_module.lr)
    assert param_group["weight_decay"] == pytest.approx(expected_wd)
    assert optimizer.defaults["betas"] == expected_betas


def test_create_optimizer_from_config_ademamix(mocked_module: BaseGraphModule) -> None:

    optimizer_cfg = OmegaConf.create(
        {
            "_target_": "anemoi.training.optimizers.AdEMAMix.AdEMAMix",
            "betas": [0.9, 0.95, 0.9999],
            "weight_decay": 0.1,
        },
    )

    expected_wd = optimizer_cfg.weight_decay
    expected_betas = tuple(optimizer_cfg.betas)

    cfg_copy = OmegaConf.create(OmegaConf.to_container(optimizer_cfg, resolve=True))

    optimizer = mocked_module._create_optimizer_from_config(cfg_copy)

    param_group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert param_group["lr"] == pytest.approx(mocked_module.lr)
    assert param_group["weight_decay"] == pytest.approx(expected_wd)
    assert optimizer.defaults["betas"] == expected_betas


def test_create_optimizer_from_config_invalid(mocked_module: BaseGraphModule) -> None:
    bad_cfg = OmegaConf.create(
        {
            "_target_": "nonexistent.OptimizerClass",
        },
    )
    with pytest.raises(Exception, match="Error locating target"):
        mocked_module._create_optimizer_from_config(bad_cfg)


def test_create_scheduler(mocked_module: BaseGraphModule) -> None:
    """Ensure cosine scheduler is constructed correctly."""
    optimizer = torch.optim.Adam(mocked_module.parameters(), lr=mocked_module.lr)
    sched_dict = mocked_module._create_scheduler(optimizer)
    scheduler = sched_dict["scheduler"]
    assert isinstance(scheduler, CosineLRScheduler)
    # Quick sanity check: scheduler should expose optimizer
    assert scheduler.optimizer is optimizer


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

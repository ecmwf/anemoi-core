# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for the EMA-of-the-weights callback.

The callback's update arithmetic is driven by hand over a tiny module, so the test needs neither a
Trainer nor a GPU.
"""

import torch
from omegaconf import OmegaConf
from torch import nn

from anemoi.training.diagnostics.callbacks import CONFIG_ENABLED_CALLBACKS
from anemoi.training.diagnostics.callbacks import _get_config_enabled_callbacks
from anemoi.training.diagnostics.callbacks.ema import EMAWeights

HALFLIFE = 4


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model.register_buffer("counter", torch.tensor(7))  # non-floating, must not be averaged
    return model


def _config(*, enabled: bool, warmup: bool = True, halflives: tuple[int, ...] = (HALFLIFE,), **kwargs) -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "ema": {
                    "enabled": enabled,
                    "halflife_steps": list(halflives),
                    "warmup": warmup,
                    "device": "cpu",
                    "log_every": 0,
                    **kwargs,
                },
            },
        },
    )


def test_registered_in_config_enabled_callbacks() -> None:
    """The callback is wired to training.ema.enabled."""
    assert ("training.ema.enabled", EMAWeights) in CONFIG_ENABLED_CALLBACKS


def test_disabled_creates_nothing() -> None:
    """(a) With the flag off, no callback is built and no shadow copy is ever allocated."""
    off = _config(enabled=False)
    assert not any(isinstance(cb, EMAWeights) for cb in _get_config_enabled_callbacks(off))

    on = _config(enabled=True)
    assert any(isinstance(cb, EMAWeights) for cb in _get_config_enabled_callbacks(on))

    # even if instantiated directly, a disabled callback allocates nothing and ignores updates
    callback = EMAWeights(off)
    assert callback._shadows == {}
    assert callback.state_dict() == {}
    callback.load_state_dict({"shadows": {"4": {}}})
    assert callback._pending_state is None


def test_no_shadow_for_integer_buffers() -> None:
    """Only floating-point parameters are averaged."""
    model = _tiny_model()
    callback = EMAWeights(_config(enabled=True))
    callback._initialise(model, source="test")
    names = set(callback._shadows[HALFLIFE])
    assert names == {name for name, p in model.named_parameters() if p.is_floating_point()}
    assert "counter" not in names


def test_warmup_equals_raw_weights_at_step_zero() -> None:
    """(b) With warmup, the EMA is the raw weights at step 0 and stays close right after it."""
    model = _tiny_model()
    callback = EMAWeights(_config(enabled=True, warmup=True, halflives=(1000,)))
    callback._initialise(model, source="test")

    # at step 0, before any update, the shadow is exactly the raw weights
    for name, param in model.named_parameters():
        torch.testing.assert_close(callback._shadows[1000][name], param.detach().to(torch.float32), rtol=0, atol=0)

    # the warmup ramp caps the decay far below the half-life decay in the first steps
    assert callback.decay_for(1000, 0) == 1.0 / 10.0
    assert callback.decay_for(1000, 90) == 91.0 / 100.0
    # and it stops biting once the ramp exceeds the half-life decay
    assert callback.decay_for(1000, 10_000_000) == 0.5 ** (1.0 / 1000)

    # without warmup the decay is the plain half-life decay from the first step
    plain = EMAWeights(_config(enabled=True, warmup=False, halflives=(1000,)))
    assert plain.decay_for(1000, 0) == 0.5 ** (1.0 / 1000)


def test_geometric_average_after_constant_steps() -> None:
    """(c) Holding the weights constant, the EMA follows the closed-form geometric decay."""
    model = _tiny_model()
    callback = EMAWeights(_config(enabled=True, warmup=False))
    callback._initialise(model, source="test")

    initial = {name: p.detach().clone() for name, p in model.named_parameters()}

    # jump the weights to a new constant value and hold them there
    target = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(1.0)
            target[name] = param.detach().clone()

    n_steps = 10
    for step in range(1, n_steps + 1):
        callback.update(model, step)

    decay = 0.5 ** (1.0 / HALFLIFE)
    remaining = decay**n_steps
    for name in initial:
        expected = remaining * initial[name] + (1.0 - remaining) * target[name]
        torch.testing.assert_close(callback._shadows[HALFLIFE][name], expected, rtol=1e-6, atol=1e-7)

    # after exactly one half-life the gap to the target is halved, which is the point of the knob
    fresh = EMAWeights(_config(enabled=True, warmup=False))
    fresh._initialise(model, source="test")
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    for step in range(1, HALFLIFE + 1):
        fresh.update(model, step)
    for name, param in model.named_parameters():
        gap = (fresh._shadows[HALFLIFE][name] - param.detach()).abs()
        torch.testing.assert_close(gap, torch.full_like(gap, 0.5), rtol=1e-6, atol=1e-6)


def test_update_every_preserves_halflife() -> None:
    """Updating every k steps with decay ** k lands in the same place as k single updates."""
    model = _tiny_model()
    single = EMAWeights(_config(enabled=True, warmup=False))
    every_two = EMAWeights(_config(enabled=True, warmup=False, update_every=2))
    single._initialise(model, source="test")
    every_two._initialise(model, source="test")

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    for step in range(1, 9):
        single.update(model, step)
    for step in range(2, 9, 2):
        every_two.update(model, step)

    for name in single._shadows[HALFLIFE]:
        torch.testing.assert_close(
            every_two._shadows[HALFLIFE][name],
            single._shadows[HALFLIFE][name],
            rtol=1e-6,
            atol=1e-7,
        )


def test_state_dict_round_trip() -> None:
    """The shadows and the step counter survive a save/load cycle."""
    model = _tiny_model()
    callback = EMAWeights(_config(enabled=True, warmup=False))
    callback._initialise(model, source="test")
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    for step in range(1, 4):
        callback.update(model, step)
    state = callback.state_dict()
    assert state["num_updates"] == 3

    restored = EMAWeights(_config(enabled=True, warmup=False))
    restored.load_state_dict(state)
    assert restored._pending_state is not None
    restored._adopt_pending_state(model)
    assert restored._num_updates == 3
    for name, tensor in callback._shadows[HALFLIFE].items():
        torch.testing.assert_close(restored._shadows[HALFLIFE][name], tensor, rtol=0, atol=0)


def test_missing_state_reinitialises_instead_of_failing() -> None:
    """An older checkpoint with no EMA state must warn, not crash."""
    callback = EMAWeights(_config(enabled=True))
    callback.load_state_dict({})
    assert callback._pending_state is None
    callback.load_state_dict({"shadows": {"999": {}}})  # half-life the config does not ask for
    assert callback._pending_state is None

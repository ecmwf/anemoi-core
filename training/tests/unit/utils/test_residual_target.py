# (C) Copyright 2026- Anemoi contributors.

import pytest
import torch

from anemoi.training.tasks import FixedOffsetsTask
from anemoi.training.utils.residual_target import compute_residual_targets


class AffineProcessor:
    def __init__(self, mean: float, stdev: float) -> None:
        self.mean = mean
        self.stdev = stdev

    def __call__(self, values: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        del in_place
        return (values - self.mean) / self.stdev


def _task(input_count: int, output_count: int) -> FixedOffsetsTask:
    input_offsets = [f"{index}h" for index in range(input_count)]
    output_offsets = [f"{index}h" for index in range(output_count)]
    return FixedOffsetsTask(input_offsets=input_offsets, output_offsets=output_offsets)


@pytest.mark.parametrize("input_count,output_count", [(1, 1), (3, 1), (2, 2), (3, 3), (4, 2)])
def test_each_output_residual_uses_its_own_statistics(input_count: int, output_count: int) -> None:
    task = _task(input_count, output_count)
    batch_size = 4096
    means = torch.arange(1, output_count + 1, dtype=torch.float32)
    stdevs = torch.arange(2, output_count + 2, dtype=torch.float32)
    residuals = torch.randn(batch_size, output_count, 1) * stdevs + means
    targets = residuals
    source = torch.zeros(batch_size, input_count, 1)

    processors = [AffineProcessor(float(mean), float(stdev)) for mean, stdev in zip(means, stdevs, strict=True)]
    normalized = compute_residual_targets(targets, source, task, processors)

    for step in range(output_count):
        assert normalized[:, step].mean().item() == pytest.approx(0.0, abs=0.08)
        assert normalized[:, step].std(unbiased=False).item() == pytest.approx(1.0, abs=0.08)


def test_residual_baselines_follow_reordered_output_offsets() -> None:
    task = FixedOffsetsTask(
        input_offsets=["-6h", "0h", "6h"],
        output_offsets=["6h", "0h", "-6h"],
    )
    source = torch.tensor([[[10.0], [20.0], [30.0]]])
    targets = torch.tensor([[[31.0], [22.0], [13.0]]])
    processor = AffineProcessor(1.0, 1.0)

    normalized = compute_residual_targets(targets, source, task, [processor] * 3)
    assert normalized.flatten().tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_single_residual_statistics_bundle_requires_explicit_broadcast() -> None:
    task = _task(2, 2)
    targets = torch.ones(16, 2, 1) * 3
    source = torch.zeros(16, 2, 1)
    processor = AffineProcessor(3.0, 2.0)

    with pytest.raises(ValueError, match="broadcast"):
        compute_residual_targets(targets, source, task, processor)

    normalized = compute_residual_targets(
        targets,
        source,
        task,
        processor,
        broadcast_single=True,
    )
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_missing_residual_statistics_fail_fast() -> None:
    task = _task(2, 2)
    targets = torch.ones(4, 2, 1)
    source = torch.zeros(4, 2, 1)
    with pytest.raises(ValueError, match="Expected 2 residual processors"):
        compute_residual_targets(targets, source, task, [AffineProcessor(0.0, 1.0)])


def test_direct_prediction_channels_use_state_statistics() -> None:
    task = _task(2, 2)
    targets = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]]).expand(64, -1, -1)
    source = torch.zeros_like(targets)
    normalized = compute_residual_targets(
        targets,
        source,
        task,
        [AffineProcessor(1.0, 1.0), AffineProcessor(2.0, 1.0)],
        state_processor=AffineProcessor(10.0, 2.0),
        direct_prediction_indices=[1],
    )

    assert torch.allclose(normalized[..., 0], torch.tensor(0.0))
    assert torch.allclose(normalized[..., 1], torch.tensor(0.0))

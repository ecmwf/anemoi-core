# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-sample operations of :class:`TabularSource` that losses, rollout and transport rely on.

``pairwise`` is what the losses reach through :func:`apply_pairwise`, ``concat_time`` is how
the forecaster's rollout rebuilds the input windows, and ``condition_shape`` sizes the transport
noise conditions. ``shard`` is only exercised on its single-process paths here; the multi-rank
``allgather`` lives in ``tests/distributed/test_tabular_source_allgather.py``.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data.utils import apply_pairwise

LAYOUT = TensorLayout(grid=0, variables=1)
ENSEMBLE_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2)
WINDOW_SPACING = 3600.0


class _FakeGroup:
    """Minimal ProcessGroup stand-in; the sharding guards only call ``size()``."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _obs(
    window_sizes: list[list[int]],
    *,
    n_variables: int = 2,
    n_members: int | None = None,
    offset: float = 0.0,
    shard_sizes: list | None = None,
    name: str = "obs",
) -> TabularSource:
    """Build an observation source with one entry of ``window_sizes`` per sample.

    Every value is distinct and traceable: sample ``s`` holds ``1000 * s + offset + 0, 1, ...``
    (plus ``0.1 * m`` for member ``m``), point ``i`` sits at ``(i, s + offset)`` and a point of
    window ``w`` has timedelta ``WINDOW_SPACING * w + i``.
    """
    data, coordinates, timedeltas, boundaries = [], [], [], []
    for sample_idx, sizes in enumerate(window_sizes):
        n_points = sum(sizes)
        values = torch.arange(n_points * n_variables, dtype=torch.float32).reshape(n_points, n_variables)
        values = values + 1000.0 * sample_idx + offset
        if n_members is not None:
            values = torch.stack([values + 0.1 * member for member in range(n_members)])
        data.append(values)
        index = torch.arange(n_points, dtype=torch.float32)
        coordinates.append(torch.stack([index, torch.full_like(index, sample_idx + offset)], dim=1))
        window_index = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes, dtype=torch.long))
        timedeltas.append(WINDOW_SPACING * window_index.float() + index)
        starts = [sum(sizes[:w]) for w in range(len(sizes))]
        boundaries.append(tuple(slice(start, start + size) for start, size in zip(starts, sizes)))
    return TabularSource(
        name=name,
        variables=[chr(ord("a") + i) for i in range(n_variables)],
        layout=LAYOUT if n_members is None else ENSEMBLE_LAYOUT,
        statistics={"mean": torch.arange(n_variables, dtype=torch.float32)},
        values=data,
        coordinates=coordinates,
        timedeltas=timedeltas,
        boundaries=boundaries,
        shard_sizes=shard_sizes,
    )


def _assert_same_payload(actual: TabularSource, expected: TabularSource) -> None:
    """Assert both sources hold the same samples, points, windows and shard sizes."""
    assert actual.batch_size == expected.batch_size
    for field_name in ("values", "coordinates", "timedeltas"):
        for sample_idx, (got, want) in enumerate(zip(getattr(actual, field_name), getattr(expected, field_name))):
            assert torch.equal(got, want), f"{field_name} of sample {sample_idx} differs"
    assert list(actual.boundaries) == list(expected.boundaries)
    assert actual.shard_sizes == expected.shard_sizes


def _sum_difference(pred: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
    """A loss that is zero for an empty sample, as the production losses are."""
    return (pred - target).sum()


class TestPairwise:
    """``pairwise`` calls the loss once per sample and averages over the non-empty samples."""

    def test_calls_the_loss_per_sample_with_aligned_pairs_and_source_metadata(self) -> None:
        pred = _obs([[2, 1], [1, 2]], offset=1.0)
        target = _obs([[2, 1], [1, 2]])
        target = target.clone(coordinates=pred.coordinates)  # same points, different values
        calls = []

        def loss(pred_sample, target_sample, scale, *, layout, statistics, name_to_index, shared):
            calls.append((pred_sample, target_sample, scale, layout, statistics, name_to_index, shared))
            return torch.tensor(float(len(calls)))

        result = pred.pairwise(target, loss, 2.0, shared="x")

        assert len(calls) == 2
        for sample_idx, (pred_sample, target_sample, scale, layout, statistics, name_to_index, shared) in enumerate(
            calls
        ):
            assert pred_sample is pred.values[sample_idx]
            assert target_sample is target.values[sample_idx]
            assert scale == 2.0
            assert layout == LAYOUT
            assert statistics is pred.statistics
            assert name_to_index == {"a": 0, "b": 1}
            assert shared == "x"
        # mean over the two samples of the per-sample results 1 and 2
        torch.testing.assert_close(result, torch.tensor(1.5))

    def test_empty_samples_do_not_dilute_the_mean(self) -> None:
        # Sample 1 has no observations; the mean is over samples 0 and 2 only.
        pred = _obs([[2], [0], [3]], offset=1.0)
        target = _obs([[2], [0], [3]]).clone(coordinates=pred.coordinates)

        result = pred.pairwise(target, _sum_difference)

        # 2 points x 2 variables x 1.0 and 3 points x 2 variables x 1.0
        torch.testing.assert_close(result, torch.tensor((4.0 + 6.0) / 2))

    def test_an_all_empty_source_returns_a_graph_connected_zero(self) -> None:
        pred = _obs([[0], [0]])
        pred = pred.map_data(lambda t: t.requires_grad_())
        target = _obs([[0], [0]])

        result = pred.pairwise(target, _sum_difference)

        assert result.item() == 0.0  # divided by max(0, 1), not by zero
        assert result.requires_grad
        result.backward()

    def test_gradient_of_each_non_empty_sample_is_scaled_by_the_number_of_non_empty_samples(self) -> None:
        pred = _obs([[2], [0], [1]]).map_data(lambda t: t.clone().requires_grad_())
        target = _obs([[2], [0], [1]])

        pred.pairwise(target, _sum_difference).backward()

        for sample in (pred.values[0], pred.values[2]):
            torch.testing.assert_close(sample.grad, torch.full_like(sample, 0.5))

    def test_per_sample_kwargs_reach_their_own_sample(self) -> None:
        pred = _obs([[2], [1], [3]])
        target = _obs([[2], [1], [3]])
        seen = []

        def loss(pred_sample, target_sample, *, weights, shared, **_):
            seen.append((pred_sample.shape[0], weights, shared))
            return torch.tensor(weights * pred_sample.shape[0])

        result = pred.pairwise(target, loss, per_sample_kwargs={"weights": [10.0, 20.0, 30.0]}, shared=1)

        assert seen == [(2, 10.0, 1), (1, 20.0, 1), (3, 30.0, 1)]
        torch.testing.assert_close(result, torch.tensor((20.0 + 20.0 + 90.0) / 3))

    def test_ensemble_predictions_pair_with_a_single_member_target(self) -> None:
        # The ensemble axis may differ (CRPS-style losses); every other axis must line up.
        pred = _obs([[2], [3]], n_members=4)
        target = _obs([[2], [3]], n_members=1)
        shapes = []

        def loss(pred_sample, target_sample, **_):
            shapes.append((tuple(pred_sample.shape), tuple(target_sample.shape)))
            return pred_sample.sum() * 0

        pred.pairwise(target, loss)

        assert shapes == [((4, 2, 2), (1, 2, 2)), ((4, 3, 2), (1, 3, 2))]

    def test_apply_pairwise_dispatches_to_the_source(self) -> None:
        pred = _obs([[2]], offset=1.0)
        target = _obs([[2]]).clone(coordinates=pred.coordinates)

        torch.testing.assert_close(apply_pairwise(pred, target, _sum_difference), torch.tensor(4.0))

    @pytest.mark.parametrize(
        ("other", "kwargs", "error", "match"),
        [
            pytest.param(
                lambda: GriddedSource(
                    name="era5",
                    variables=["a", "b"],
                    layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
                    values=torch.zeros(1, 1, 1, 2, 2),
                    coordinates=torch.zeros(2, 2),
                ),
                {},
                TypeError,
                "must be a TabularSource",
                id="gridded-other",
            ),
            pytest.param(lambda: _obs([[2], [1]], n_members=1), {}, ValueError, "same layout", id="layout"),
            pytest.param(lambda: _obs([[2]]), {}, ValueError, "same number of samples", id="sample-count"),
            pytest.param(
                lambda: _obs([[2], [1]]),
                {"weights": 1.0, "per_sample_kwargs": {"weights": [1.0, 2.0]}},
                ValueError,
                "both shared and per-sample",
                id="shared-and-per-sample",
            ),
            pytest.param(
                lambda: _obs([[2], [1]]),
                {"per_sample_kwargs": {"weights": [1.0]}},
                ValueError,
                r"'weights' requires one value per sample \(2\)",
                id="per-sample-length",
            ),
        ],
    )
    def test_rejects_misaligned_sources_before_computing(self, other, kwargs, error, match) -> None:
        loss = Mock()
        with pytest.raises(error, match=match):
            _obs([[2], [1]]).pairwise(other(), loss, **kwargs)
        loss.assert_not_called()

    def test_rejects_samples_of_different_sizes(self) -> None:
        with pytest.raises(AssertionError, match="Sample 1 of both views must have the same shape"):
            _obs([[2], [1]]).pairwise(_obs([[2], [2]]), _sum_difference)

    def test_rejects_samples_at_different_points(self) -> None:
        with pytest.raises(AssertionError, match="Sample 0 of both views must have the same coordinates"):
            _obs([[2]]).pairwise(_obs([[2]], offset=1.0), _sum_difference)

    def test_rejects_a_source_without_samples(self) -> None:
        empty = TabularSource(
            name="obs", variables=["a"], layout=LAYOUT, values=[], coordinates=[], timedeltas=[], boundaries=[]
        )
        with pytest.raises(ValueError, match="empty sparse source view"):
            empty.pairwise(empty, _sum_difference)


class TestConcatTime:
    """``concat_time`` appends whole time windows, sample by sample, as the rollout advance does."""

    def test_split_windows_rejoin_to_the_original(self) -> None:
        source = _obs([[2, 3], [1, 0], [0, 2]])

        rejoined = source.select_time([0]).concat_time(source.select_time([1]))

        _assert_same_payload(rejoined, source)
        assert rejoined.time_size == 2

    def test_windows_are_appended_in_argument_order_with_shifted_boundaries(self) -> None:
        first = _obs([[2], [1]])
        second = _obs([[3, 1], [0, 2]], offset=500.0)

        joined = first.concat_time(second)

        assert joined.time_size == 3
        assert joined.boundaries == [
            (slice(0, 2), slice(2, 5), slice(5, 6)),
            (slice(0, 1), slice(1, 1), slice(1, 3)),
        ]
        for sample_idx in range(2):
            assert torch.equal(joined.values[sample_idx], torch.cat([first.values[sample_idx], second.values[sample_idx]]))
            assert torch.equal(
                joined.timedeltas[sample_idx],
                torch.cat([first.timedeltas[sample_idx], second.timedeltas[sample_idx]]),
            )
        _assert_same_payload(joined.select_time([0]), first)
        _assert_same_payload(joined.select_time([1, 2]), second)

    def test_reordered_windows_follow_the_new_order(self) -> None:
        # The rollout moves a later input window to an earlier slot.
        source = _obs([[2, 1], [1, 3]])

        swapped = source.select_time([1]).concat_time(source.select_time([0]))

        _assert_same_payload(swapped, source.select_time([1, 0]))
        assert swapped.boundaries == [(slice(0, 1), slice(1, 3)), (slice(0, 3), slice(3, 4))]
        assert torch.equal(swapped.timedeltas[0], torch.tensor([WINDOW_SPACING + 2.0, 0.0, 1.0]))

    def test_more_than_two_sources_and_no_sources(self) -> None:
        source = _obs([[1, 2, 3], [2, 0, 1]])
        windows = [source.select_time([t]) for t in range(3)]

        _assert_same_payload(windows[0].concat_time(*windows[1:]), source)
        _assert_same_payload(source.concat_time(), source)

    def test_concatenates_ensemble_data_along_the_grid_axis(self) -> None:
        source = _obs([[2, 3], [1, 1]], n_members=2)

        rejoined = source.select_time([0]).concat_time(source.select_time([1]))

        _assert_same_payload(rejoined, source)
        assert rejoined.values[0].shape == (2, 5, 2)

    def test_extends_per_window_shard_sizes(self) -> None:
        # One rank's view of a source split over two ranks: shard sizes are per sample and window.
        source = _obs([[2, 1], [1, 1]], shard_sizes=[[[2, 1], [1, 1]], [[1, 0], [1, 1]]])

        rejoined = source.select_time([0]).concat_time(source.select_time([1]))

        assert rejoined.shard_sizes == source.shard_sizes
        _assert_same_payload(rejoined, source)

    @pytest.mark.parametrize(
        ("other", "match"),
        [
            pytest.param(lambda: _obs([[1]]), "batch size", id="batch-size"),
            pytest.param(lambda: _obs([[1], [1]], n_variables=3), "variables", id="variables"),
            pytest.param(lambda: _obs([[1], [1]], n_members=1), "layout", id="layout"),
            pytest.param(lambda: _obs([[1], [1]], shard_sizes=[[[1]], [[1]]]), "sharding", id="sharding"),
        ],
    )
    def test_rejects_windows_of_another_structure(self, other, match) -> None:
        with pytest.raises(ValueError, match=f"mismatched .*{match}"):
            _obs([[1], [1]]).concat_time(other())

    def test_rejects_a_different_ensemble_size(self) -> None:
        with pytest.raises(ValueError, match="ensemble size"):
            _obs([[1]], n_members=2).concat_time(_obs([[1]], n_members=3))

    def test_rejects_gridded_windows(self) -> None:
        gridded = GriddedSource(
            name="era5",
            variables=["a", "b"],
            layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
            values=torch.zeros(1, 1, 1, 2, 2),
            coordinates=torch.zeros(2, 2),
        )
        with pytest.raises(TypeError, match="Cannot append GriddedSource windows"):
            _obs([[1]]).concat_time(gridded)


class TestConditionShape:
    """``condition_shape`` is ``(batch, 1, ensemble, 1, 1)`` whatever the per-sample point counts."""

    def test_ragged_samples_without_an_ensemble_axis(self) -> None:
        assert _obs([[5], [0], [2, 7]]).condition_shape == (3, 1, 1, 1, 1)

    def test_ragged_samples_with_members(self) -> None:
        assert _obs([[5], [1]], n_members=3).condition_shape == (2, 1, 3, 1, 1)

    def test_a_condition_of_that_shape_reaches_every_sample_and_member(self) -> None:
        source = _obs([[2], [3]], n_members=2)
        condition = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(source.condition_shape)

        scaled = source.map_with_condition(lambda data, sigma: data * sigma, condition)

        for sample_idx, sample in enumerate(source.values):
            for member in range(2):
                factor = condition[sample_idx, 0, member, 0, 0]
                torch.testing.assert_close(scaled.values[sample_idx][member], sample[member] * factor)

    def test_rejects_a_source_without_samples(self) -> None:
        empty = TabularSource(
            name="obs", variables=["a"], layout=LAYOUT, values=[], coordinates=[], timedeltas=[], boundaries=[]
        )
        with pytest.raises(ValueError, match="empty sparse data list"):
            _ = empty.condition_shape

    def test_rejects_inconsistent_ensemble_sizes(self) -> None:
        source = _obs([[1], [1]], n_members=2)
        source = source.clone(values=[source.values[0], source.values[1].repeat(2, 1, 1)])
        with pytest.raises(ValueError, match="Inconsistent ensemble sizes"):
            _ = source.condition_shape


class TestShard:
    """Tabular sources are sharded by the reader; ``shard`` only passes them through."""

    @pytest.mark.parametrize("group", [None, _FakeGroup(1)], ids=["no-group", "single-rank"])
    def test_without_a_distributed_group_is_a_noop(self, group) -> None:
        source = _obs([[2], [1]])
        assert source.shard(group) is source
        template = source.template()
        assert template.shard(group) is template

    def test_an_already_sharded_source_is_returned_unchanged(self) -> None:
        source = _obs([[2]], shard_sizes=[[[2, 1]]])
        assert source.shard(_FakeGroup(2)) is source
        template = source.template()
        assert template.shard(_FakeGroup(2)) is template

    def test_a_replicated_source_cannot_be_sharded_over_several_ranks(self) -> None:
        source = _obs([[2]])
        with pytest.raises(NotImplementedError, match="sharded at read time"):
            source.shard(_FakeGroup(2))
        with pytest.raises(NotImplementedError, match="sharded at read time"):
            source.template().shard(_FakeGroup(2))


def test_allgather_rejects_shard_sizes_of_another_group_before_any_collective() -> None:
    # Two-rank window sizes gathered over a three-rank group: rejected up front, on the second
    # sample, so no partial sequence of per-window collectives is issued.
    source = _obs([[2], [2]], shard_sizes=[[[2, 1, 0]], [[1, 1]]])
    with pytest.raises(ValueError, match=r"sample 1, window 0"):
        source.allgather(_FakeGroup(3))


def test_tree_lists_the_window_sizes_of_every_sample() -> None:
    tree = _obs([[2, 3], [1, 0]], n_members=4).tree(prefix="> ")

    assert tree.label == "> obs | TabularSource[torch.float32, cpu]"
    assert [child.label for child in tree.children] == [
        "Dim 0 (batch): 2",
        "Dim 1 (ensemble): 4",
        "Dim 2 (grid): 2 time slices",
        "\tSample 1: 5 <- 2 + 3",
        "\tSample 2: 1 <- 1 + 0",
        "Dim 3 (variables): 2",
    ]

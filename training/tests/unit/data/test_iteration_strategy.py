# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np

from anemoi.training.data.iteration_strategy import CrossDatasetIterationStrategy


def make_dataset(
    valid_date_indices: dict[str, np.ndarray],
    chunk_index_range: dict[str, np.ndarray],
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> SimpleNamespace:
    return SimpleNamespace(
        valid_date_indices=valid_date_indices,
        chunk_index_range=chunk_index_range,
        shuffle=shuffle,
        rng=np.random.default_rng(seed),
        label="test",
        worker_id=0,
        global_rank=0,
        model_comm_group_id=0,
        model_comm_group_rank=0,
        sample_comm_group_id=0,
    )


def test_cross_dataset_iteration_strategy_preserves_domain_order_across_sample_groups() -> None:
    valid_date_indices = {"dataset_a": np.arange(8), "dataset_b": np.arange(4)}
    group_0_ranges = {"dataset_a": np.arange(0, 4), "dataset_b": np.arange(0, 2)}
    group_1_ranges = {"dataset_a": np.arange(4, 8), "dataset_b": np.arange(2, 4)}

    strategy = CrossDatasetIterationStrategy()
    group_0 = strategy._sample_indices(make_dataset(valid_date_indices, group_0_ranges))
    group_1 = strategy._sample_indices(make_dataset(valid_date_indices, group_1_ranges))

    assert [domain for domain, _ in group_0] == [domain for domain, _ in group_1]
    for domain in valid_date_indices:
        group_0_indices = {index for sampled_domain, index in group_0 if sampled_domain == domain}
        group_1_indices = {index for sampled_domain, index in group_1 if sampled_domain == domain}
        assert group_0_indices.isdisjoint(group_1_indices)


def test_cross_dataset_iteration_strategy_without_shuffle_preserves_domain_and_index_order() -> None:
    dataset = make_dataset(
        {"dataset_a": np.arange(4), "dataset_b": np.arange(3)},
        {"dataset_a": np.arange(1, 3), "dataset_b": np.arange(0, 2)},
        shuffle=False,
    )
    strategy = CrossDatasetIterationStrategy()

    assert strategy._sample_indices(dataset) == [
        ("dataset_a", 1),
        ("dataset_a", 2),
        ("dataset_b", 0),
        ("dataset_b", 1),
    ]


def test_cross_dataset_iteration_strategy_repeats_for_same_seed() -> None:
    valid_date_indices = {"dataset_a": np.arange(8), "dataset_b": np.arange(4)}
    chunk_index_range = {"dataset_a": np.arange(0, 4), "dataset_b": np.arange(0, 2)}

    strategy = CrossDatasetIterationStrategy()
    first = make_dataset(valid_date_indices, chunk_index_range)
    second = make_dataset(valid_date_indices, chunk_index_range)

    assert strategy._sample_indices(first) == strategy._sample_indices(second)

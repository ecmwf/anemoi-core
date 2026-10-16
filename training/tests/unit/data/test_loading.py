# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset

from anemoi.training.data.loading import CrossDatasetBatchSampler


def make_sampler(
    sizes: tuple[int, ...],
    *,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 1,
) -> CrossDatasetBatchSampler:
    dataset = ConcatDataset([TensorDataset(torch.arange(size)) for size in sizes])
    return CrossDatasetBatchSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        drop_last=True,
    )


def test_cross_dataset_sampler_preserves_domain_order_across_sample_groups() -> None:
    group_0 = list(make_sampler((8, 4), num_replicas=2, rank=0))
    group_1 = list(make_sampler((8, 4), num_replicas=2, rank=1))

    assert [batch[0] >= 8 for batch in group_0] == [batch[0] >= 8 for batch in group_1]
    for offset, end in ((0, 8), (8, 12)):
        group_0_indices = {index for batch in group_0 for index in batch if offset <= index < end}
        group_1_indices = {index for batch in group_1 for index in batch if offset <= index < end}
        assert group_0_indices.isdisjoint(group_1_indices)


def test_cross_dataset_sampler_without_shuffle_preserves_domain_and_index_order() -> None:
    sampler = make_sampler((4, 3), shuffle=False, batch_size=2)

    assert list(sampler) == [[0, 1], [4, 5], [2, 3], [6]]


def test_cross_dataset_sampler_repeats_for_same_seed_and_epoch() -> None:
    first = make_sampler((8, 4), seed=123, batch_size=2)
    second = make_sampler((8, 4), seed=123, batch_size=2)
    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


def test_cross_dataset_sampler_batches_are_domain_homogeneous() -> None:
    batches = make_sampler((8, 5), batch_size=3)

    assert all(all(index < 8 for index in batch) or all(index >= 8 for index in batch) for batch in batches)


def test_cross_dataset_sampler_matches_single_domain_sampler_order() -> None:
    sampler = make_sampler((8,), shuffle=False, batch_size=3)

    assert list(sampler) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_cross_dataset_sampler_resumes_at_each_batch_boundary() -> None:
    expected = list(make_sampler((6, 4), shuffle=False, batch_size=2))

    for stop_after in range(len(expected) + 1):
        sampler = make_sampler((6, 4), shuffle=False, batch_size=2)
        iterator = iter(sampler)
        consumed = [next(iterator) for _ in range(stop_after)]
        resumed = make_sampler((6, 4), shuffle=False, batch_size=2)
        resumed.load_state_dict(sampler.state_dict())

        assert consumed + list(resumed) == expected

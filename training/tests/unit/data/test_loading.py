# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset

from anemoi.training.data.loading import DataLoadingPolicy
from anemoi.training.data.loading import MultiDomainBatchSampler
from anemoi.training.data.loading import MultiDomainLoadingPolicy
from anemoi.training.data.loading import SynchronizedLoadingPolicy


def make_sampler(
    sizes: tuple[int, ...],
    *,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 1,
) -> MultiDomainBatchSampler:
    dataset = ConcatDataset([TensorDataset(torch.arange(size)) for size in sizes])
    return MultiDomainBatchSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        drop_last=True,
    )


def test_multi_domain_sampler_preserves_domain_order_across_sample_groups() -> None:
    group_0 = list(make_sampler((8, 4), num_replicas=2, rank=0))
    group_1 = list(make_sampler((8, 4), num_replicas=2, rank=1))

    assert [batch[0] >= 8 for batch in group_0] == [batch[0] >= 8 for batch in group_1]
    for offset, end in ((0, 8), (8, 12)):
        group_0_indices = {index for batch in group_0 for index in batch if offset <= index < end}
        group_1_indices = {index for batch in group_1 for index in batch if offset <= index < end}
        assert group_0_indices.isdisjoint(group_1_indices)


def test_multi_domain_sampler_without_shuffle_preserves_domain_and_index_order() -> None:
    sampler = make_sampler((4, 3), shuffle=False, batch_size=2)

    assert list(sampler) == [[0, 1], [2, 3], [4, 5], [6]]


def test_multi_domain_sampler_repeats_for_same_seed_and_epoch() -> None:
    first = make_sampler((8, 4), seed=123, batch_size=2)
    second = make_sampler((8, 4), seed=123, batch_size=2)
    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


def test_multi_domain_sampler_shuffles_equal_sized_domains_differently() -> None:
    batches = list(make_sampler((16, 16)))
    first_domain = [index for batch in batches for index in batch if index < 16]
    second_domain = [index - 16 for batch in batches for index in batch if index >= 16]

    assert sorted(first_domain) == sorted(second_domain) == list(range(16))
    assert first_domain != second_domain


def test_multi_domain_sampler_batches_are_domain_homogeneous() -> None:
    batches = make_sampler((8, 5), batch_size=3)

    assert all(all(index < 8 for index in batch) or all(index >= 8 for index in batch) for batch in batches)


def test_multi_domain_sampler_matches_single_domain_sampler_order() -> None:
    sampler = make_sampler((8,), shuffle=False, batch_size=3)

    assert list(sampler) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_multi_domain_sampler_resumes_at_each_batch_boundary() -> None:
    expected = list(make_sampler((6, 4), batch_size=2))

    for stop_after in range(len(expected) + 1):
        sampler = make_sampler((6, 4), batch_size=2)
        iterator = iter(sampler)
        consumed = [next(iterator) for _ in range(stop_after)]
        resumed = make_sampler((6, 4), batch_size=2)
        resumed.load_state_dict(sampler.state_dict())

        assert consumed + list(resumed) == expected


@pytest.mark.parametrize("num_replicas", [1, 2])
def test_multi_domain_sampler_resumes_at_each_batch_boundary_after_first_epoch(num_replicas: int) -> None:
    def build(*, finish_first_epoch: bool = False) -> MultiDomainBatchSampler:
        sampler = make_sampler((12, 8), num_replicas=num_replicas, batch_size=2)
        if finish_first_epoch:
            # leaves every domain sampler at its end-of-epoch position
            list(sampler)
        sampler.set_epoch(1)
        return sampler

    expected = list(build())

    for stop_after in range(len(expected) + 1):
        sampler = build(finish_first_epoch=True)
        iterator = iter(sampler)
        consumed = [next(iterator) for _ in range(stop_after)]
        resumed = build()
        resumed.load_state_dict(sampler.state_dict())

        assert consumed + list(resumed) == expected


DATASETS = ["era5", "cerra", "synop"]


def encoders(strategy: str | None, sources: list[str]) -> dict:
    encoder = {"source_datasets": sources}
    if strategy is not None:
        encoder["dataset_fusing_strategy"] = strategy
    return {"shared": encoder, "obs": {"source_datasets": ["synop"], "dataset_fusing_strategy": "none"}}


def test_co_loaded_datasets_per_policy() -> None:
    assert SynchronizedLoadingPolicy().co_loaded_datasets(DATASETS) == [frozenset(DATASETS)]
    assert MultiDomainLoadingPolicy().co_loaded_datasets(DATASETS) == [frozenset({name}) for name in DATASETS]


@pytest.mark.parametrize("strategy", ["none", None])
def test_synchronized_policy_rejects_unfused_encoder_with_several_sources(strategy: str | None) -> None:
    with pytest.raises(ValueError, match=r"Encoder 'shared'.*SynchronizedLoadingPolicy.*\['era5', 'cerra'\]"):
        SynchronizedLoadingPolicy().check_encoder_fusion(encoders(strategy, ["era5", "cerra"]), DATASETS)


def test_multi_domain_policy_accepts_unfused_encoder_with_several_sources() -> None:
    MultiDomainLoadingPolicy().check_encoder_fusion(encoders("none", ["era5", "cerra"]), DATASETS)


def test_encoder_fusion_check_accepts_sequential_and_single_sources() -> None:
    policy = SynchronizedLoadingPolicy()
    policy.check_encoder_fusion(encoders("sequential", ["era5", "cerra"]), DATASETS)
    policy.check_encoder_fusion(encoders("none", ["era5"]), DATASETS)


def test_encoder_fusion_check_ignores_sources_that_are_not_loaded() -> None:
    SynchronizedLoadingPolicy().check_encoder_fusion(encoders("none", ["era5", "unloaded"]), DATASETS)


def test_custom_policy_defaults_to_one_batch_for_all_datasets() -> None:
    class CustomPolicy(DataLoadingPolicy):
        def build_dataset(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def build_batch_sampler(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    assert CustomPolicy().co_loaded_datasets(DATASETS) == [frozenset(DATASETS)]
    with pytest.raises(ValueError, match="CustomPolicy"):
        CustomPolicy().check_encoder_fusion(encoders("none", ["era5", "cerra"]), DATASETS)

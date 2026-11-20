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
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytest_mock import MockFixture
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset

from anemoi.training.data.loading import GroupedDatasetLoadingPolicy
from anemoi.training.data.loading import MultiDomainBatchSampler
from anemoi.training.data.loading import MultiDomainLoadingPolicy
from anemoi.training.data.loading import SynchronizedLoadingPolicy
from anemoi.training.schemas.dataloader import DataLoadingPolicySchema


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


@pytest.mark.parametrize(
    ("target", "extra_fields", "expected_type"),
    [
        (SynchronizedLoadingPolicy, {}, SynchronizedLoadingPolicy),
        (MultiDomainLoadingPolicy, {}, MultiDomainLoadingPolicy),
        (
            GroupedDatasetLoadingPolicy,
            {"dataset_groups": [["dataset_a", "dataset_b"], ["dataset_c"]]},
            GroupedDatasetLoadingPolicy,
        ),
    ],
)
def test_loading_policy_schema_instantiates_each_policy_target(
    target: type,
    extra_fields: dict,
    expected_type: type,
) -> None:
    schema = DataLoadingPolicySchema(target_=f"{target.__module__}.{target.__name__}", **extra_fields)

    config = OmegaConf.create(schema.model_dump(by_alias=True))
    policy = instantiate(config)

    assert isinstance(policy, expected_type)
    for key, value in extra_fields.items():
        assert getattr(policy, key) == value


def test_grouped_policy_stores_groups_and_combinations() -> None:
    dataset_groups = [["dataset_a", "dataset_b"], ["dataset_c", "dataset_d"]]

    policy = GroupedDatasetLoadingPolicy(OmegaConf.create(dataset_groups))
    assert policy.dataset_groups == dataset_groups
    assert policy.combinations == [
        ("dataset_a", "dataset_c"),
        ("dataset_a", "dataset_d"),
        ("dataset_b", "dataset_c"),
        ("dataset_b", "dataset_d"),
    ]


@pytest.mark.parametrize("dataset_groups", [[[]], [["dataset_a"], []]])
def test_grouped_policy_rejects_empty_groups(dataset_groups: list[list[str]]) -> None:
    with pytest.raises(ValueError, match="must not contain an empty group"):
        GroupedDatasetLoadingPolicy(OmegaConf.create(dataset_groups))


def test_grouped_policy_rejects_datasets_in_multiple_groups() -> None:
    with pytest.raises(ValueError, match="must belong to a single group"):
        GroupedDatasetLoadingPolicy(OmegaConf.create([["dataset_a", "dataset_b"], ["dataset_b"]]))


def test_grouped_policy_builds_one_dataset_per_group_combination(mocker: MockFixture) -> None:
    dataset_groups = [["dataset_a", "dataset_b"], ["dataset_c", "dataset_d"]]
    policy = GroupedDatasetLoadingPolicy(OmegaConf.create(dataset_groups))
    data_readers = {name: mocker.MagicMock() for group in dataset_groups for name in group}
    relative_date_indices = {name: [0, 2, 6] for name in data_readers}
    mock_multi_dataset = mocker.patch(
        "anemoi.training.data.loading.MultiDataset",
        side_effect=lambda readers, **_kwargs: mocker.MagicMock(data_readers=readers),
    )

    dataset = policy.build_dataset(data_readers, relative_date_indices, fake_dataloading=True)

    expected_combinations = [
        ("dataset_a", "dataset_c"),
        ("dataset_a", "dataset_d"),
        ("dataset_b", "dataset_c"),
        ("dataset_b", "dataset_d"),
    ]
    assert [tuple(domain.data_readers) for domain in dataset.datasets] == expected_combinations
    assert set(dataset.data_readers) == set(data_readers)
    assert [call.args[1] for call in mock_multi_dataset.call_args_list] == [
        {name: relative_date_indices[name] for name in combination} for combination in expected_combinations
    ]
    assert all(call.kwargs["fake_dataloading"] is True for call in mock_multi_dataset.call_args_list)


def test_grouped_policy_requires_exactly_the_configured_datasets(mocker: MockFixture) -> None:
    policy = GroupedDatasetLoadingPolicy(OmegaConf.create([["dataset_a"], ["dataset_b"]]))
    data_readers = {"dataset_a": mocker.MagicMock()}

    with pytest.raises(ValueError, match="must contain exactly the configured datasets"):
        policy.build_dataset(data_readers, {"dataset_a": [0]})


def test_grouped_policy_check_dataset_compatibility_rejects_datasets_with_different_variable_orders(
    mocker: MockFixture,
) -> None:
    policy = GroupedDatasetLoadingPolicy(OmegaConf.create([["dataset_a", "dataset_b"]]))
    dataset = mocker.MagicMock(
        name_to_index={
            "dataset_a": {"temperature": 0, "pressure": 1},
            "dataset_b": {"pressure": 0, "temperature": 1},
        },
    )

    with pytest.raises(ValueError, match="must have the same variables in the same order"):
        policy.check_dataset_compatibility(dataset)


def test_grouped_policy_check_dataset_compatibility_accepts_datasets_with_same_variable_order(
    mocker: MockFixture,
) -> None:
    policy = GroupedDatasetLoadingPolicy(OmegaConf.create([["dataset_a", "dataset_b"]]))
    dataset = mocker.MagicMock(
        name_to_index={
            "dataset_a": {"temperature": 0, "pressure": 1},
            "dataset_b": {"temperature": 0, "pressure": 1},
        },
        data={
            "dataset_a": mocker.MagicMock(typed_variables={}),
            "dataset_b": mocker.MagicMock(typed_variables={}),
        },
    )

    policy.check_dataset_compatibility(dataset)


def build_grouped_dataset(
    mocker: MockFixture,
    sizes: tuple[int, ...],
) -> tuple[GroupedDatasetLoadingPolicy, ConcatDataset]:
    policy = GroupedDatasetLoadingPolicy(OmegaConf.create([["dataset_a", "dataset_b"], ["dataset_c", "dataset_d"]]))
    sizes_by_combination = dict(zip(policy.combinations, sizes, strict=True))
    data_readers = {name: mocker.MagicMock() for group in policy.dataset_groups for name in group}
    relative_date_indices = {name: [0] for name in data_readers}

    def make_domain(readers: dict, _indices: dict, **_kwargs: object) -> TensorDataset:
        domain = TensorDataset(torch.arange(sizes_by_combination[tuple(readers)]))
        domain.data_readers = readers
        return domain

    mocker.patch("anemoi.training.data.loading.MultiDataset", side_effect=make_domain)
    return policy, policy.build_dataset(data_readers, relative_date_indices)


def test_grouped_policy_batch_sampler_batches_each_combination_in_order(mocker: MockFixture) -> None:
    policy, dataset = build_grouped_dataset(mocker, (8, 4, 6, 2))

    sampler = policy.build_batch_sampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=0,
        batch_size=2,
        drop_last=False,
    )

    assert isinstance(sampler, MultiDomainBatchSampler)
    assert list(sampler) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
    ]


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

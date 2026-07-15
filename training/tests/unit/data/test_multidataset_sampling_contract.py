# (C) Copyright 2026- Anemoi contributors.

from dataclasses import dataclass

import numpy as np

from anemoi.training.data.multidataset import MultiDataset


@dataclass
class FakeReader:
    anchors: np.ndarray

    num_sequences: int = 2

    def compute_anchors(self, relative_indices: list[int]) -> np.ndarray:
        del relative_indices
        return self.anchors


def _make_dataset() -> MultiDataset:
    anchors = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
        dtype=np.int64,
    )
    return MultiDataset({"data": FakeReader(anchors)}, {"data": [0]})


def _collect_indices(dataset: MultiDataset, *, epoch: int, groups: int, workers: int) -> list[int]:
    dataset.set_epoch(epoch)
    dataset.sample_comm_num_groups = groups
    collected: list[int] = []
    for group_id in range(groups):
        dataset.sample_comm_group_id = group_id
        for worker_id in range(workers):
            dataset.per_worker_init(workers, worker_id)
            permutation = dataset.rng.choice(
                dataset.valid_date_indices,
                size=len(dataset.valid_date_indices),
                replace=False,
            )
            collected.extend(permutation[dataset.chunk_index_range].tolist())
    return collected


def test_all_valid_anchors_are_sampled_once_across_groups_and_workers(monkeypatch) -> None:
    monkeypatch.setattr("anemoi.training.data.multidataset.get_base_seed", lambda: 1234)
    dataset = _make_dataset()

    sampled = _collect_indices(dataset, epoch=0, groups=3, workers=2)

    assert len(sampled) == len(dataset.anchors)
    assert sorted(sampled) == list(range(len(dataset.anchors)))
    assert len(set(sampled)) == len(sampled)


def test_sampling_is_reproducible_per_epoch_and_reshuffles_between_epochs(monkeypatch) -> None:
    monkeypatch.setattr("anemoi.training.data.multidataset.get_base_seed", lambda: 1234)
    first = _collect_indices(_make_dataset(), epoch=0, groups=3, workers=2)
    repeat = _collect_indices(_make_dataset(), epoch=0, groups=3, workers=2)
    next_epoch = _collect_indices(_make_dataset(), epoch=1, groups=3, workers=2)

    assert first == repeat
    assert first != next_epoch


def test_unequal_valid_anchor_counts_do_not_drop_remainder(monkeypatch) -> None:
    monkeypatch.setattr("anemoi.training.data.multidataset.get_base_seed", lambda: 7)
    anchors = np.array([[0, step] for step in range(11)], dtype=np.int64)
    dataset = MultiDataset({"data": FakeReader(anchors, num_sequences=1)}, {"data": [0]})

    sampled = _collect_indices(dataset, epoch=0, groups=4, workers=3)

    assert sorted(sampled) == list(range(11))

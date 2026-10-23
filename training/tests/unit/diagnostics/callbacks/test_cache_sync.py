# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from types import SimpleNamespace

from anemoi.training.diagnostics.callbacks.cache_sync import CacheSyncCallback


class FakeCache:
    def __init__(self):
        self.sync_count = 0

    def update_global_view(self):
        self.sync_count += 1


def make_trainer(*, epoch=0, sanity_checking=False):
    return SimpleNamespace(
        current_epoch=epoch,
        sanity_checking=sanity_checking,
        state=SimpleNamespace(fn="fit"),
        global_rank=0,
        datamodule=None,
    )


def test_syncs_once_before_validation():
    cache = FakeCache()
    callback = CacheSyncCallback(cache=cache)
    trainer = make_trainer()

    callback.on_validation_start(trainer, None)
    callback.on_validation_start(trainer, None)

    assert cache.sync_count == 1


def test_does_not_sync_during_sanity_check():
    cache = FakeCache()
    callback = CacheSyncCallback(cache=cache)

    callback.on_validation_start(make_trainer(sanity_checking=True), None)

    assert cache.sync_count == 0


def test_respects_epoch_cadence():
    cache = FakeCache()
    callback = CacheSyncCallback(cache=cache, sync_every_n_epochs=2)

    callback.on_validation_start(make_trainer(epoch=0), None)
    callback.on_validation_start(make_trainer(epoch=1), None)

    assert cache.sync_count == 1


def test_does_not_sync_after_validation():
    cache = FakeCache()
    callback = CacheSyncCallback(cache=cache)
    trainer = make_trainer(epoch=0)

    callback.on_validation_start(trainer, None)
    callback.on_train_epoch_end(trainer, None)

    assert cache.sync_count == 1

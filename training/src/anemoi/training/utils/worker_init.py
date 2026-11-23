# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from torch.utils.data import get_worker_info

LOGGER = logging.getLogger(__name__)


def set_blosc_threads(blosc_threads: int | None) -> None:
    """Set the number of blosc threads in this process.

    numcodecs uses blosc threads only in the main thread of the main process. In a dataloader
    worker, blosc uses one thread. This function lets blosc use ``blosc_threads`` threads in the
    worker, for the decompression of zarr chunks and for the dataset cache.

    Parameters
    ----------
    blosc_threads : int | None
        Number of blosc threads. If None, do not change the numcodecs settings.

    """
    if blosc_threads is None:
        return
    import numcodecs.blosc

    numcodecs.blosc.use_threads = True
    # set the number again in the worker: the thread pool of the parent process is not in the fork
    numcodecs.blosc.set_nthreads(blosc_threads)


def worker_init_func(worker_id: int, blosc_threads: int | None = None) -> None:
    """Configures each dataset worker process.

    Calls per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID
    blosc_threads : int | None, optional
        Number of blosc threads in the worker. If None, blosc uses one thread.

    Raises
    ------
    RuntimeError
        If worker_info is None

    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    set_blosc_threads(blosc_threads)
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )

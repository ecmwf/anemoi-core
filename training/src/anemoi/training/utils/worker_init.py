# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
import warnings


def set_blosc_threads(blosc_threads: int | None) -> None:
    """Set the number of blosc threads in this process.

    numcodecs uses blosc threads only in the main thread of the main process. In a dataloader
    worker, blosc uses one thread. This function lets blosc use ``blosc_threads`` threads in the
    worker, for the decompression of zarr chunks and for the dataset cache.

    In a process that is a fork of the process that imported numcodecs, numcodecs always uses one
    thread, also when ``use_threads`` is True. The reason is that the blosc thread pool of the
    parent process is not in the fork. Thus, in a fork, this function starts blosc again and tells
    numcodecs that this process imported it.

    Parameters
    ----------
    blosc_threads : int | None
        Number of blosc threads. If None, do not change the numcodecs settings.

    """
    if blosc_threads is None:
        return
    import numcodecs.blosc

    if os.getpid() != numcodecs.blosc._importer_pid:
        # a fork: start blosc again, so that this process has its own thread pool
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            numcodecs.blosc.destroy()
            numcodecs.blosc.init()
        numcodecs.blosc._importer_pid = os.getpid()
    numcodecs.blosc.use_threads = True
    numcodecs.blosc.set_nthreads(blosc_threads)


def worker_init_func(worker_id: int, blosc_threads: int | None = None) -> None:
    """Configure a dataloader worker process.

    Parameters
    ----------
    worker_id : int
        Worker ID. This function does not use it.
    blosc_threads : int | None, optional
        Number of blosc threads in the worker. If None, blosc uses one thread.

    """
    del worker_id
    set_blosc_threads(blosc_threads)

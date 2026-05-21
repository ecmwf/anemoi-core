# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from numcodecs.blosc import get_nthreads
from numcodecs.blosc import set_nthreads
from torch.utils.data import get_worker_info

LOGGER = logging.getLogger(__name__)


def worker_init_func(worker_id: int, blosc_num_threads: int | None = None) -> None:
    """Configures each dataset worker process.

    Calls per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID
    blosc_num_threads : int, optional
        Number of threads to use for Blosc decompression, by default None.
        If None, the number of threads will be determined by Blosc.

    Raises
    ------
    RuntimeError
        If worker_info is None

    """
    # read the dataloader config to set the number of threads blosc should use when decompressing zarr chunks.
    if blosc_num_threads is not None:
        set_nthreads(blosc_num_threads)
        LOGGER.info("Set number of threads used by Blosc to %d.", blosc_num_threads)
    else:
        LOGGER.info("Using Blosc default number of threads (%d) for decompression.", get_nthreads())

    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )

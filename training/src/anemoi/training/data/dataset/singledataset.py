# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import random
from collections.abc import Callable
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices
import cProfile
from concurrent.futures import ThreadPoolExecutor

LOGGER = logging.getLogger(__name__)

#TODO fix issue where cache is recreated each epoch

class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        """
        self.label = label

        self.data = data_reader

        self.timestep = timestep
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

        # relative index of dates to extract
        self.relative_date_indices = relative_date_indices
        
        use_cache=True
        if use_cache:
            self.cache=DatasetCache(
                cache_root=str(os.getenv("TMPDIR")),
                dataset_path="/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr",
                ds=self.data,
                )
        else:
            self.cache = None

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    def __getitem__(self, index) -> np.ndarray:
        if self.cache is not None:
            #check the cache first, the cache will fallback to self.ds in the case of a miss
            return self.cache[index]
        else:
            return self.data[index]

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i.
        """
        return get_usable_indices(
                self.data.missing,
                len(self.data),
                np.array(self.relative_date_indices, dtype=np.int64),
                self.data.trajectory_ids,
                )

    def set_comm_group_info(
            self,
            global_rank: int,
            model_comm_group_id: int,
            model_comm_group_rank: int,
            model_comm_num_groups: int,
            reader_group_rank: int,
            reader_group_size: int,
            ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
                "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
                "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
                global_rank,
                model_comm_group_id,
                model_comm_group_rank,
                model_comm_num_groups,
                reader_group_rank,
                )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        shard_end = (self.sample_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
                "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
                worker_id,
                os.getpid(),
                self.global_rank,
                self.model_comm_group_id,
                low,
                high,
                )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.info(
                (
                    "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                    "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
                    ),
                worker_id,
                self.label,
                os.getpid(),
                self.global_rank,
                self.model_comm_group_id,
                self.model_comm_group_rank,
                self.sample_comm_group_id,
                base_seed,
                sanity_rnd,
                )

    #def get(self, index):
    #    result = self.cache[index]
    #    if result is None:
    #        result= self.data[index]
    #        self.cache[index] = result
    #    return result

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                    self.valid_date_indices,
                    size=len(self.valid_date_indices),
                    replace=False,
                    )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
                (
                    "Worker pid %d, label %s, worker id %d, global_rank %d, "
                    "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
                    ),
                os.getpid(),
                self.label,
                self.worker_id,
                self.global_rank,
                self.model_comm_group_id,
                self.model_comm_group_rank,
                self.sample_comm_group_id,
                shuffled_chunk_indices[:10],
                )

        for i in shuffled_chunk_indices:
            start = i + self.relative_date_indices[0]
            end = i + self.relative_date_indices[-1] + 1
            timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
            # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
            # data[start...] will be replaced with data[self.relative_date_indices + i]

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x = self[start:end:timeincrement, :, :, grid_shard_indices] # check cache

            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x = self[start:end:timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
        """



from pathlib import Path
import torch.distributed as dist
import http.server
import socketserver
import shutil
import zarr
from zarr.convenience import PathNotFoundError
import pytorch_lightning as pl
import socket
import threading
from functools import partial
class DatasetCache():
    """
        self.cache = DatasetCache($TMPDIR, /home/mlx/...)

        result = self.cache[start:end:timeincrement, :, :, :]
        if result is None:
            result = self.data[start:end:timeincrement, :, :, :]
            self.cache[start:end:timeincrement, :, :, :] = result
        return result
    """
    def __init__(self, cache_root, dataset_path, ds):
        self.cache_root = Path(cache_root)
        self.dataset_path=Path(dataset_path)
        self.ds = ds

        self.is_initalised = False


        #option to write async
        self.write_async=False
        if self.write_async:
            self.chunk_writer = ThreadPoolExecutor(max_workers=4) #used to sync write chunks into cache, compressing is very slow ~3s for 200MB

    #split between init and setup to match distirbuted strategy from PL structure (we need the proccess group to be initalised)
    def setup(self) -> bool:

        #use default global proc group if proc_group is none
        #TODO you should be able to use it if you only run on a single gpu
        if not dist.is_initialized(): #TODO and check proc_group is valid
            LOGGER.info("Torch distributed is not initalised, can't use distributed cache")
            return False

        if self.is_initalised:
            return True

        self.proc_group=dist.new_group(backend="cpu:gloo",group_desc="cache_proc_group")

        self.rank = dist.get_rank(self.proc_group)
        self.local_rank = int(os.environ.get("SLURM_LOCALID", -1)) 

        self.device = "cpu" #torch.device(f"cuda:{self.local_rank if self.local_rank != -1 else self.rank}" if torch.cuda.is_available() else "cpu")
        self.world_size = dist.get_world_size(self.proc_group)
        self.hostnames = self._get_all_hostnames()

        #data_dir = Path("/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8.zarr")

        #creates space under self.cache_dir
        self.filesystem= f"{self.dataset_path}" #TODO read from zarr metadata

        # once the dataset is loaded, this will become an array of len(dates) where the value corresponds to which processes cache a file is in
        # -1 => filesystem
        self.global_cache_registry=None
        self.cache_registry=None
        self._init_cache()

        self.root_port = 7100
        self.port = self.root_port + self.rank 
        if self.local_rank == 0:
            self._start_server(self.cache_root, self.port) #need to use cache root so that other local caches can be found

        LOGGER.info(f"{self.rank=}: Initalised a cache under {self.cache_path}")
        self.is_initalised=True
        return True


    @staticmethod
    def create_uncompressed_cache_array(src_array_path, dst_cache_path, array_name="data"):
        """
        Create a Zarr array at dst_cache_path using metadata from src_array_path,
        but disable compression (compressor=None).
        """
        import json

        src_array_path = Path(src_array_path)
        dst_cache_path = Path(dst_cache_path)

        # --- Read .zarray metadata from source ---
        zarray_file = src_array_path / ".zarray"
        if not zarray_file.exists():
            raise FileNotFoundError(f"No .zarray metadata found in {src_array_path}")

        with open(zarray_file, "r") as f:
            meta = json.load(f)

        shape     = tuple(meta["shape"])
        chunks    = tuple(meta["chunks"])
        dtype     = meta["dtype"]
        order     = meta.get("order", "C")
        fill      = meta.get("fill_value", None)
        filters   = meta.get("filters", None)

        # --- Create Zarr group for cache destination ---
        cache_group = zarr.open(dst_cache_path, mode="a")

        # --- Create dataset with same metadata but compressor=None ---
        cache_array = cache_group.create_dataset(
            array_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            order=order,
            fill_value=fill,
            filters=filters,
            compressor=None,
            overwrite=True,
        )

        return cache_array


    def _init_cache(self, delete_existing_cache=False):

        if self.cache_root is None:
            LOGGER.info("Cache root not given, defaulting to $TMPDIR")
            self.cache_root=Path(os.getenv("TMPDIR"))

        #create space under cache_root for cache
        assert self.cache_root.exists()
        self.cache_path = Path(f"{self.cache_root}/cache_rank_{self.rank}")
        if self.cache_path.exists() and delete_existing_cache:
            LOGGER.info(f"Existing cache found under {self.cache_path}. Deleting because {delete_existing_cache=}")
            shutil.rmtree(self.cache_path) #this breaks restarting after different epochs
        self.cache_path.mkdir(exist_ok=True)

        #copy zarr metadata from filesystem
        # This is needed so we can load chunks from the cache like we would from the remote zarr
        # BUT we will have a lot of gaps in the local copy, so we will have a seperate list self.cache_registry of which elements are present or not
        #TODO replace with zarr.empty_like()
        #shutil.copy2(f"{self.filesystem}/data/.zarray", self.cache_path)
        #shutil.copy2(f"{self.filesystem}/data/.zattrs", self.cache_path) #some datasets dont have, not sure if its needed tbh
        #shutil.copy2(f"{self.filesystem}/.zgroup", self.cache_path)

        disable_compression=True
        if disable_compression:
            self.cache = self.create_uncompressed_cache_array(f"{self.filesystem}/data", self.cache_path)

        else:
            #from numcodecs import Blosc

            #compressor = Blosc(cname='zstd', clevel=0, shuffle=Blosc.NOSHUFFLE)
            #has an error on 2md epoch bc path isnt empty
            self.cache = zarr.open_like(self.ds, self.cache_path, mode="w") #, compression_opts=compressor) #ignored
        #self.cache = zarr.open(self.cache_path, mode="w")

        dates = 100 #len(self) 
        #dates = self.ds.size
        #dates = self.ds.ds_train.data.shape[0]

        if self.cache_registry is None:
            self.cache_registry=torch.zeros(dates, dtype=torch.int32, device=self.device)
            self.cache_registry[:] = -1

        self.remote_cache_roots=self._get_all_cache_roots()


    def _start_server(self, directory, port):
        #directory = Path("/").resolve()
        directory = Path(self.cache_root).resolve()
        handler=http.server.SimpleHTTPRequestHandler

        def no_logging(*args):
            return
        handler.log_message=no_logging #by default the http server LOGGER.infos a lot to stdout, this silences it

        handler = partial(handler, directory=str(directory))

        try: 
            LOGGER.info(f"{self.rank=}: Starting a server hosting {directory} on http://{self._get_hostname(self.rank)}:{port}")
            httpd = socketserver.TCPServer(("", port), handler)
            self.thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            self.thread.start()
            #Need to either kill this thread after each epoch or kill and restart
        except OSError:
            LOGGER.info(f"{self.rank=}: {port=} already in use, naively assuming its an exisitng server and carrying on")


    def _get_hostname(self, rank):
        return self.hostnames[rank]

    def _get_all_hostnames(self):
        """Gather the hostname of every rank into a list (length = world_size)."""
        my_host = socket.gethostname()

        # Gather all hostnames as Python objects
        hostnames = [None for _ in range(self.world_size)]
        dist.all_gather_object(hostnames, my_host, self.proc_group)
        LOGGER.info(f"{self.rank=} {hostnames=}")
        return hostnames

    def _get_all_cache_roots(self):
        """Gather the cache roots of every rank into a list (length = world_size).

        needed bc $TMPDIR is different paths on different ranks
        """
        root = self.cache_root

        # Gather all hostnames as Python objects
        roots = [None for _ in range(self.world_size)]
        dist.all_gather_object(roots, root, self.proc_group)
        return roots

    def _shutdown_server(self):
        try: 
            if self.server_metadata is not None:
                httpd, thread = self.server_metadata
                httpd.shutdown()
                thread.join()
        except AttributeError:
            pass

    def __del__(self):
        self._shutdown_server()

    def update_global_view(self) -> None:
        """ Communicate with other procs and share who has what files in their caches. Should be called after an epoch."""
        all_gather_buffer = [torch.zeros_like(self.cache_registry) for _ in range(self.world_size)]
        dist.all_gather(all_gather_buffer, self.cache_registry, self.proc_group)
        self.global_cache_registry = torch.stack(all_gather_buffer)


    def _get_remote_cache_url(self, remote_rank):
        remote_host = self._get_hostname(remote_rank)
        port = self.root_port + remote_rank
        #remote_cache_path = f"{self.remote_cache_roots[remote_rank]}/cache_rank_{remote_rank}" #just need relative path here, will be aded to the root dir of our http server (e.g. $TMPDIR)
        remote_cache_path = f"cache_rank_{remote_rank}" #just need relative path here, will be aded to the root dir of our http server (e.g. $TMPDIR)
        #import pdb
        #breakpoint()
        return f"http://{remote_host}:{port}/{remote_cache_path}"

    def check_cache(self, date) -> list[int]:
        """ checks either local or global registry. returns list of procs containing file in their SSD"""
        #if self.global_cache_registry is not None:
        #    cache_subset=self.global_cache_registry[:,date] # (num_proc) array, -1 if not in cache, non-negative if in cache
        #else:
        #    assert self.cache_registry is not None
        #    cache_subset = [self.cache_registry[date]]
        if self.global_cache_registry is not None:
            # Slice from torch tensor → convert to 1D numpy array
            cache_subset = self.global_cache_registry[:, date].detach().cpu().numpy().ravel()
        else:
            # Local cache registry is already a Python mapping
            cache_subset = self.cache_registry[date].detach().cpu().numpy()

        print(f"{cache_subset=}")

        # Convert numpy values to Python int and filter out -1
        cache_hits = [int(x) for x in cache_subset if int(x) != -1]
        #x is len(dates) e.g. 3 for rollout 1
        #cache_hits = [x for x in cache_subset if (x != -1).all()]
        return list(cache_hits)

    #try to copy without uncompressing and recompressing
    #doesnt work bc ds isnt a zarr
    #@staticmethod
    #def copy_chunk_raw(src_array, dst_array, index):
    #    src_store = src_array.store
    #    dst_store = dst_array.store

    #    chunk_key = src_array._chunk_key(index)
    #    dst_store[chunk_key] = src_store[chunk_key]

    #TODO make it so that it can return partial matches from cache and the rest from filesystem
    # e,g, cache_subset=array([ 0,  0, -1], dtype=int32); self.rank=0: cache_hits=[0, 0]
    def __getitem__(self, index) -> np.ndarray:
        """ Reads cache regsitry, based on result fetches file from local SSD, remote SSD or None if its a miss"""

        verbose=False


        if not self.is_initalised:
            success = self.setup()
            if not success:
                return self.ds[index]

        cache_hits = self.check_cache(index[0])
        #LOGGER.info(f"{self.rank=}: {cache_hits=}")
        print(f"{self.rank=}: {cache_hits=}")

        #with cProfile.Profile() as pr:
        if len(cache_hits) == 0:
            #if cache_hits.numel() == 0:
            #Cache miss
            data = self.ds[index]
            if self.write_async:
                self.chunk_writer.submit(lambda cache, index, data: cache[index].set(data), self.cache, index, data)

            else:
                self.cache[index] = data #VERY slow ~3s for 2MB chunks #what i truly need is a way to copy compressed chunks directly from filesystem to cache
                #self.copy_chunk_raw(self.ds, self.cache, index)
            if verbose:
                LOGGER.info(f"{self.rank=}: cache miss on date {index=}")
            
        #elif (cache_hits == self.rank).any():
        elif self.rank in cache_hits:
            #Cache hit on local SSD
            
            data = self.cache[index]
            
            if verbose:
                LOGGER.info(f"{self.rank=}: local cache hit on date {index}")
                
        else:
            #cache hit on remote SSD
            
            #take the first cache hit
            remote_rank = cache_hits[0] # TODO could be smarter by picking local ranks 
            remote_cache_url=self._get_remote_cache_url(remote_rank)
            LOGGER.info(f"{self.rank=} accessing {index=} on rank {remote_rank} at {remote_cache_url=}")
            
            try:
                data = zarr.open(remote_cache_url, mode="r")[index]
                #TODO figure out why i get cache misses
                if verbose:
                    LOGGER.info(f"{self.rank=}: remote cache hit (rank {remote_rank}) on date {index}")
            except (PathNotFoundError, KeyError) as e:
                LOGGER.info(f"Error loading remote date {index} from {remote_rank} to {self.rank}. full error: {e}. falling back to filesystem.")
                data = self.ds[index]
        #pr.print_stats()
            
        return data

    def __setitem__(self, key, value):
        if self.is_initalised:
            #add item to cache
            self.cache[key] = value
            #update caches bookkeeping
            date = key[0]
            self.cache_registry[date] = self.rank
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch == 0:
            self.update_global_view()
        

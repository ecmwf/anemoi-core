import os
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
import http.server
import socketserver
from functools import partial
from multiprocessing import Value

from anemoi.datasets import open_dataset

import zarr
from zarr.convenience import PathNotFoundError
from pathlib import Path
import threading

import numpy as np
import socket
import subprocess


import pytorch_lightning as pl
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

import logging
LOGGER = logging.getLogger(__name__)


class CachedDataWrapper:
    """Wrapper that intercepts data access and routes through cache."""
    
    def __init__(self, original_data, cache_instance):
        self.original_data = original_data
        self.cache = cache_instance
        self._access_count = 0
        LOGGER.info(f"CachedDataWrapper created for {type(original_data).__name__}")
        
    def __getitem__(self, index):
        """Intercept array access and route through cache."""
        self._access_count += 1
        if self._access_count <= 5:
            LOGGER.debug(f"CachedDataWrapper.__getitem__ called (access #{self._access_count}) with index type: {type(index).__name__}, value: {index if not isinstance(index, (tuple, slice)) or len(str(index)) < 50 else '...'}")
        
        # Handle different index types
        if isinstance(index, tuple):
            # Multi-dimensional indexing like data[time, :, :, grid]
            time_idx = index[0]
            rest_idx = index[1:] if len(index) > 1 else ()
            
            if isinstance(time_idx, slice):
                # Slice access - fetch multiple time steps
                start, stop, step = time_idx.indices(len(self.original_data))
                LOGGER.debug(f"CachedDataWrapper: Fetching slice [{start}:{stop}:{step}] (total {(stop-start)//max(1,step or 1)} samples)")
                results = []
                for i in range(start, stop, step or 1):
                    single_result = self.cache.fetch(i, verbose=False)
                    if rest_idx:
                        single_result = single_result[rest_idx]
                    results.append(single_result)
                return np.stack(results, axis=0)
            elif isinstance(time_idx, int):
                # Single time index
                LOGGER.debug(f"CachedDataWrapper: Fetching single time index {time_idx}")
                result = self.cache.fetch(time_idx, verbose=False)
                if rest_idx:
                    result = result[rest_idx]
                return result
            else:
                # Fallback to original data access
                LOGGER.warning(f"CachedDataWrapper: Unhandled tuple index type: {type(time_idx).__name__}, falling back to original data")
                return self.original_data[index]
        elif isinstance(index, (int, np.integer)):
            # Single integer index
            LOGGER.debug(f"CachedDataWrapper: Fetching single integer index {index}")
            return self.cache.fetch(int(index), verbose=False)
        elif isinstance(index, slice):
            # Simple slice
            start, stop, step = index.indices(len(self.original_data))
            LOGGER.debug(f"CachedDataWrapper: Fetching simple slice [{start}:{stop}:{step}]")
            results = []
            for i in range(start, stop, step or 1):
                results.append(self.cache.fetch(i, verbose=False))
            return np.stack(results, axis=0)
        else:
            # Fallback for other index types
            LOGGER.warning(f"CachedDataWrapper: Unhandled index type: {type(index).__name__}, falling back to original data")
            return self.original_data[index]
    
    def __len__(self):
        return len(self.original_data)
    
    def __getattr__(self, name):
        """Delegate attribute access to original data."""
        return getattr(self.original_data, name)


#TODO change to overwriting the dataset insetad of data module?
class DatasetCache(AnemoiDatasetsDataModule):
    def __init__(self, ds, cache_root, dataset_path, proc_group=None):
        self.ds=ds
        self.cache_root = Path(cache_root)
        self.dataset_path=dataset_path
        self.proc_group = proc_group
        
        # For multi-dataset scenarios, we cache the first dataset by default
        # In the future, this could be made configurable
        self.dataset_names = None
        self.primary_dataset_name = None
        
        self.is_initalised = False
        
        # Cache statistics - use shared memory for cross-process visibility
        self.cache_hits_local = Value('i', 0)  # 'i' = signed int
        self.cache_hits_remote = Value('i', 0)
        self.cache_misses = Value('i', 0)
        self.total_fetches = Value('i', 0)
        
        # Store the wrapped dataset to prevent it from being recreated
        self._cached_ds_train = None
        
    @property
    def ds_train(self):
        """Return the training dataset with cache wrapper applied."""
        if self._cached_ds_train is None:
            self._cached_ds_train = self.ds.ds_train
        return self._cached_ds_train
        
    def __getattr__(self, name):
        """
        Called *only* if the attribute wasn't found on self.
        Delegate to the underlying dataset (self.ds).
        """
        return getattr(self.ds, name)
    

    #split between init and setup to match distirbuted strategy from PL structure (we need the proccess group to be initalised)
    def setup(self, **kwargs):
        
        #use default global proc group if proc_group is none
        #TODO you should be able to use it if you only run on a single gpu
        if not dist.is_initialized(): #TODO and check proc_group is valid
            raise ValueError("Torch distributed is not initalised, can't use distributed cache")
                
        self.rank = dist.get_rank(self.proc_group)
        #self.local_rank = int(os.environ.get("SLURM_LOCALID", (int(os.environ.get("LOCAL_RANK", 0)))))
        
        #self.device = torch.device(f"cuda:{self.local_rank if self.local_rank != -1 else self.rank}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        
        # Set the CUDA device before any collective operations (required for NCCL)
        if torch.cuda.is_available() and self.device.type == "cuda":
            #TODO(cathal) fix silly hack which will only work on single nodes
            torch.cuda.set_device(self.rank)
            LOGGER.info(f"Rank {self.rank}: Set CUDA device to {self.rank}")
        
        self.world_size = dist.get_world_size(self.proc_group)
        self.hostnames = self._get_all_hostnames()

        # Determine which dataset(s) we're working with
        # For MultiDataset, cache the first dataset by default
        # Use self.ds_train which uses our cached property
        self.dataset_names = list(self.ds_train.datasets.keys())
        self.primary_dataset_name = self.dataset_names[0]
        LOGGER.info(f"DatasetCache: Found datasets {self.dataset_names}, using '{self.primary_dataset_name}' for zarr metadata")
        
        #creates space under self.cache_dir
        self.filesystem= f"{self.dataset_path}" #TODO read from zarr metadata
        
        # once the dataset is loaded, this will become an array of len(dates) where the value corresponds to which processes cache a file is in
        # -1 => filesystem
        self.global_cache_registry=None
        self.cache_registry=None
        self._init_cache()
       
        self.root_port = 8000
        self.port = self.root_port + self.rank 
        self._start_server(self.cache_root, self.port) #need to use cache root so that other local caches can be found
        
        # Inject cache wrapper into the underlying datasets
        self._inject_cache_wrapper()
        
        LOGGER.info(f"{self.rank=}: Initalised a cache under {self.cache_path}")
        self.is_initalised=True
    
    def _inject_cache_wrapper(self):
        """Replace the underlying dataset's data accessor with cached version."""
        # Use self.ds_train which accesses our cached property that stores the wrapped instance
        train_dataset = self.ds_train
        
        for dataset_name, dataset in train_dataset.datasets.items():
            if dataset_name == self.primary_dataset_name:
                original_data = dataset.data
                LOGGER.info(f"Rank {self.rank}: Injecting cache wrapper for dataset '{dataset_name}' (type: {type(original_data).__name__})")
                dataset.data = CachedDataWrapper(original_data, self)
                # Verify injection
                if isinstance(dataset.data, CachedDataWrapper):
                    LOGGER.info(f"Rank {self.rank}: Cache wrapper injected successfully for '{dataset_name}' - data is now {type(dataset.data).__name__}")
                else:
                    LOGGER.error(f"Rank {self.rank}: Cache wrapper injection FAILED for '{dataset_name}' - data is still {type(dataset.data).__name__}")
            else:
                LOGGER.info(f"Rank {self.rank}: Skipping cache injection for dataset '{dataset_name}' (only caching primary dataset)")
        
        # Double-check after injection
        LOGGER.info(f"Rank {self.rank}: Verifying injection after completion:")
        for dataset_name, dataset in train_dataset.datasets.items():
            LOGGER.info(f"Rank {self.rank}:   Dataset '{dataset_name}' data type: {type(dataset.data).__name__}")
       
    def _init_cache(self, delete_existing_cache=True):
        
        if self.cache_root is None:
            LOGGER.info("Cache root not given, defaulting to $TMPDIR")
            self.cache_root=Path(os.getenv("TMPDIR"))
        
        #create space under cache_root for cache
        assert self.cache_root.exists()
        self.cache_path = Path(f"{self.cache_root}/cache_rank_{self.rank}")
        if self.cache_path.exists() and delete_existing_cache:
            LOGGER.info(f"Existing cache found under {self.cache_path}. Deleting because {delete_existing_cache=}")
            shutil.rmtree(self.cache_path)
        self.cache_path.mkdir(exist_ok=True)
        
        #copy zarr metadata from filesystem
        # This is needed so we can load chunks from the cache like we would from the remote zarr
        # BUT we will have a lot of gaps in the local copy, so we will have a seperate list self.cache_registry of which elements are present or not
        #TODO replace with zarr.empty_like()
        shutil.copy2(f"{self.filesystem}/data/.zarray", self.cache_path)
        #shutil.copy2(f"{self.filesystem}/data/.zattrs", self.cache_path) #some datasets dont have, not sure if its needed tbh
        shutil.copy2(f"{self.filesystem}/.zgroup", self.cache_path)
        
        self.cache = zarr.open(self.cache_path, mode="w")
       
        dates = len(self) 
        #dates = self.ds.size
        #dates = self.ds.ds_train.data.shape[0]
        
        if self.cache_registry is None:
            # Keep cache_registry on CPU with shared memory for DataLoader worker access
            self.cache_registry = torch.zeros(dates, dtype=torch.int32, device='cpu').share_memory_()
            self.cache_registry[:] = -1
            LOGGER.info(f"Rank {self.rank}: Created cache registry on CPU with shared memory for {dates} dates")
            
        self.remote_cache_roots=self._get_all_cache_roots()
        
    
    def _start_server(self, directory, port):
        #directory = Path("/").resolve()
        directory = Path(self.cache_root).resolve()
        handler=http.server.SimpleHTTPRequestHandler
        
        def no_logging(*args):
            return
        handler.log_message=no_logging #by default the http server LOGGER.infos a lot to stdout, this silences it
        
        handler = partial(handler, directory=str(directory))
        
        # Allow socket reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        
        # Store server metadata for proper shutdown
        self.httpd = httpd
        self.server_thread = thread
        
        LOGGER.info(f"Serving {directory} on http://{self._get_hostname(self.rank)}:{port}")
        
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
        """Shutdown the HTTP server and close the socket."""
        try:
            if hasattr(self, 'httpd') and self.httpd is not None:
                LOGGER.info(f"Rank {self.rank}: Shutting down HTTP server on port {self.port}")
                self.httpd.shutdown()
                if hasattr(self, 'server_thread') and self.server_thread is not None:
                    self.server_thread.join(timeout=5)
                # Close the socket to free the port
                self.httpd.server_close()
                LOGGER.info(f"Rank {self.rank}: HTTP server shut down successfully")
        except Exception as e:
            LOGGER.warning(f"Rank {self.rank}: Error shutting down server: {e}")

    def __del__(self):
       self._shutdown_server()
            
    def update_global_view(self) -> None:
        """ Communicate with other procs and share who has what files in their caches. Should be called after an epoch."""
        all_gather_buffer = [torch.zeros_like(self.cache_registry) for _ in range(self.world_size)]
        dist.all_gather(all_gather_buffer, self.cache_registry, self.proc_group)
        self.global_cache_registry = torch.stack(all_gather_buffer).share_memory_()  # Keep on CPU with shared memory
        LOGGER.info(f"Rank {self.rank}: Global cache registry updated. Cache now aware of {(self.global_cache_registry != -1).sum().item()} cached items across all ranks")
    
    def print_cache_stats(self) -> None:
        """Print cache statistics."""
        total = self.total_fetches.value

        #LOGGER.info(f"Rank {self.rank}: CACHE MISS on date {date} (total: hits_local={self.cache_hits_local}, hits_remote={self.cache_hits_remote}, misses={self.cache_misses})")
        if total == 0:
            LOGGER.info(f"Rank {self.rank}: No cache accesses yet")
            return
        
        hit_rate_local = (self.cache_hits_local.value / total) * 100 if total > 0 else 0
        hit_rate_remote = (self.cache_hits_remote.value / total) * 100 if total > 0 else 0
        miss_rate = (self.cache_misses.value / total) * 100 if total > 0 else 0
        
        LOGGER.info(f"Rank {self.rank}: Cache Statistics:")
        LOGGER.info(f"  Total fetches: {total}")
        LOGGER.info(f"  Local hits: {self.cache_hits_local.value} ({hit_rate_local:.1f}%)")
        LOGGER.info(f"  Remote hits: {self.cache_hits_remote.value} ({hit_rate_remote:.1f}%)")
        LOGGER.info(f"  Misses: {self.cache_misses.value} ({miss_rate:.1f}%)")
        LOGGER.info(f"  Items in local cache: {(self.cache_registry != -1).sum().item()}")
        
            
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
        if self.global_cache_registry is not None:
            # Slice from torch tensor → convert to 1D numpy array
            cache_subset = self.global_cache_registry[:, date].detach().cpu().numpy().ravel()
        else:
            # Local cache registry is a torch tensor
            cache_subset = [self.cache_registry[date].item()]

        # Convert numpy values to Python int and filter out -1
        cache_hits = [int(x) for x in cache_subset if int(x) != -1]
        return cache_hits

    def fetch(self, date, verbose=False) -> np.ndarray:
        """ Reads cache regsitry, based on result fetches file from local SSD, remote SSD or filesytem"""

        #LOGGER.info(f"Rank {self.rank}: Fetching date {date} (verbose={verbose})")
        
        self.total_fetches.value += 1
        cache_hits = self.check_cache(date)
        
        # Get the primary dataset's raw data for caching
        # Use self.ds_train to access our cached property
        primary_dataset = self.ds_train.datasets[self.primary_dataset_name]
        primary_data = primary_dataset.data  # Underlying dataset (e.g., Zarr)
        
        # Check if we're accessing the wrapper itself (avoid infinite recursion)
        if isinstance(primary_data, CachedDataWrapper):
            primary_data = primary_data.original_data
        
        if len(cache_hits) == 0:
        #if cache_hits.numel() == 0:
            #Cache miss, go to filesystem
            
            self.cache_misses.value += 1
            data = primary_data[date]
            
            # add data to local cache
            #TODO can i do this async while i return data
            self.cache[date] = data
            self.cache_registry[date] = self.rank
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: CACHE MISS on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
            
        elif self.rank in cache_hits:
            #Cache hit on local SSD
            
            self.cache_hits_local.value += 1
            data = self.cache[date]            
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: LOCAL CACHE HIT on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
                
        else:
            #cache hit on remote SSD
            
            self.cache_hits_remote.value += 1
            #take the first cache hit
            remote_rank = cache_hits[0] # TODO could be smarter by picking local ranks 
            remote_cache_url=self._get_remote_cache_url(remote_rank)
            LOGGER.info(f"Rank {self.rank}: accessing date={date} on rank {remote_rank} at {remote_cache_url=}")
            
            try:
                data = zarr.open(remote_cache_url, mode="r")[date]
                #TODO figure out why i get cache misses
            except (PathNotFoundError, KeyError) as e:
                LOGGER.info(f"Error loading remote date {date} from {remote_rank} to {self.rank}. full error: {e}. falling back to filesystem.")
                data = primary_data[date]
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: REMOTE CACHE HIT (rank {remote_rank}) on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
            
        return data

    def __iter__(self) -> np.ndarray:
        if not self.is_initalised:
            self.setup()
        
        dates=len(self)
            
        for date in range(dates):
            #LOGGER.info(f"Rank {self.rank}: Iterating date {date}/{dates}")
            yield self.fetch(date)
        
    def __getitem__(self, index):
        # allow integer or slice access
        if isinstance(index, slice):
            return [self.fetch(i) for i in range(*index.indices(len(self)))]
        else:
            return self.fetch(index)

    def __len__(self):
        # MultiDataset.valid_date_indices contains all valid time indices
        # Use self.ds_train to access our cached property
        return len(self.ds_train.valid_date_indices)
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch == 0:
            self.update_global_view()
        
if __name__ == "__main__":
    #run with 'torchrun --nproc-per-node 2 remote_cache.py'
    dataset_path=Path("/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8.zarr")
    cache_dir=Path(os.getenv("TMPDIR"))
    ds = open_dataset(dataset_path)
    cached_ds = DatasetCache(ds, cache_root=cache_dir, dataset_path=dataset_path)
    rank=cached_ds.rank
    start=rank * 10
    end = start + 10
    LOGGER.info(f"rank {rank}: loading from {start} to {end}")
    for x in range(start, end):
        cached_ds[x] 
        dist.barrier()
        #LOGGER.info(f"rank {rank}: loaded date {x}: {cached_ds[x]}")
        
    cached_ds.update_global_view()

    # for the 2nd round, half data will be seen locally, other half will be seen remotely
    start = start + 5
    end = end + 5
    LOGGER.info(f"rank {rank}: loading from {start} to {end}")
    #while True:
    for x in range(start, end):
        cached_ds[x] 
        dist.barrier()
        #if rank == 0:
        #    LOGGER.info(f"rank {rank}: loaded date {x}: {cached_ds[x]}")
        
    #import pdb
    #breakpoint()
    cached_ds._shutdown_server()

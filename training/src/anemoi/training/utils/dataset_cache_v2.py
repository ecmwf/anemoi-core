import os
import errno
import fcntl
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
import threading

import numpy as np
import socket


import pytorch_lightning as pl
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

import logging
LOGGER = logging.getLogger(__name__)


class CachedDataWrapper:
    """Wrapper that intercepts data access and routes through cache."""
    
    def __init__(self, original_data, cache_instance):
        self.original_data = original_data
        self.cache = cache_instance
        LOGGER.info(f"CachedDataWrapper created for {type(original_data).__name__}")
        
    def __getitem__(self, index):
        """Intercept array access and route through cache."""
        
        # Handle different index types
        if isinstance(index, tuple):
            # Multi-dimensional indexing like data[time, :, :, grid]
            time_idx = index[0]
            rest_idx = index[1:] if len(index) > 1 else ()
            
            if isinstance(time_idx, slice):
                # Slice access - fetch multiple time steps
                start, stop, step = time_idx.indices(len(self.original_data))
                results = []
                for i in range(start, stop, step or 1):
                    single_result = self.cache.fetch(i, verbose=False)
                    if rest_idx:
                        single_result = single_result[rest_idx]
                    results.append(single_result)
                return np.stack(results, axis=0)
            elif isinstance(time_idx, int):
                # Single time index
                result = self.cache.fetch(time_idx, verbose=False)
                if rest_idx:
                    result = result[rest_idx]
                return result
            else:
                # Fallback to original data access
                return self.original_data[index]
        elif isinstance(index, (int, np.integer)):
            # Single integer index
            return self.cache.fetch(int(index), verbose=False)
        elif isinstance(index, slice):
            # Simple slice
            start, stop, step = index.indices(len(self.original_data))
            results = []
            for i in range(start, stop, step or 1):
                results.append(self.cache.fetch(i, verbose=False))
            return np.stack(results, axis=0)
        else:
            # Fallback for other index types
            return self.original_data[index]

    def __len__(self):
        return len(self.original_data)
    
    def __getattr__(self, name):
        """Delegate attribute access to original data."""
        return getattr(self.original_data, name)


#TODO change to overwriting the dataset insetad of data module?
class DatasetCache(AnemoiDatasetsDataModule):
    def __init__(self, ds, cache_root, dataset_path, proc_group=None, hostname_suffix=None):
        self.ds=ds # the dataset object to be cached
        self.cache_root = Path(cache_root) # the path where the cache will be stored (e.g. $TMPDIR)
        self.dataset_path=dataset_path # the path to the original dataset on the shared filesystem (e.g. zarr store path)
        self.proc_group = proc_group # torch distributed process group. The cached datasets will be shared among ranks in the same process group. If None, use the default global group.
        
        self.is_initalised = False

        # optional suffix which will be appended to hostnames
        self.hostname_suffix=hostname_suffix
        
        # Store the wrapped dataset to prevent it from being recreated
        self._cached_ds_train = None

        # For multi-dataset scenarios, we cache the first dataset by default
        # TODO extend to support caching multiple datasets, currently the code assumes there's only one dataset and uses its metadata for the cache structure
        self.dataset_names = self.ds.dataset_names
        self.primary_dataset_name = None

        # store access stats
        self.stats = {"local": 0, "remote": 0, "miss": 0}
        self.total_fetches = 0
        
    @property
    def ds_train(self):
        """Return the training dataset with cache wrapper applied."""
        if self._cached_ds_train is None:
            self._cached_ds_train = self.ds.ds_train
        return self._cached_ds_train
        
    def __getattr__(self, name):
        """
        Delegate __getattr__ to the underlying dataset (self.ds).
        Called *only* if the attribute wasn't found on self.
        """
        return getattr(self.ds, name)
    

    #split between init and setup to match distirbuted strategy from PL structure (we need the proccess group to be initalised)
    def setup(self, **kwargs):
        """ Set up the distributed cache. 
        
        This includes determining the local rank and process group sizes and resolving hostnames of remote processes.
        
        Node leaders (lowest global rank on each node) will additionally create the nodes cache directory and launch the HTTP server to handle remote cache accesses.
        """
        
        #use default global proc group if proc_group is none
        #TODO you should be able to use it if you only run on a single gpu
        if not dist.is_initialized(): #TODO and check proc_group is valid
            raise ValueError("Torch distributed is not initalised, can't use distributed cache")
                
        self.proc_group = dist.new_group(ranks=None, backend="gloo") # create a new process group to select the hostnames, this is required because we need gloo backend for this
        self.rank = dist.get_rank(self.proc_group)
        self.world_size = dist.get_world_size(self.proc_group)
        
        gpus_per_node = 4 #TODO read from system config
        self.device = torch.device(f"cuda:{self.rank % gpus_per_node}" if torch.cuda.is_available() else "cpu")
        
        # Set the CUDA device before any collective operations (required for NCCL)
        if torch.cuda.is_available() and self.device.type == "cuda":
            #TODO(cathal) fix silly hack which will only work on single nodes
            torch.cuda.set_device(self.rank % gpus_per_node)
            LOGGER.info(f"Rank {self.rank}: Set CUDA device to {self.rank % gpus_per_node}")
        
        self.node_id = self.rank // gpus_per_node
        self.num_nodes = self.world_size // gpus_per_node 
        self.is_node_leader = (self.rank % gpus_per_node) == 0
        LOGGER.info(f"Rank {self.rank}: node_id={self.node_id}, is_node_leader={self.is_node_leader}")

        self.hostnames = self._get_all_hostnames(suffix=self.hostname_suffix)

        # Determine which dataset(s) we're working with
        # For MultiDataset, cache the first dataset by default
        # Use self.ds_train which uses our cached property
        self.dataset_names = list(self.ds_train.datasets.keys())
        self.primary_dataset_name = self.dataset_names[0]
        LOGGER.info(f"DatasetCache: Found datasets {self.dataset_names}, using '{self.primary_dataset_name}' for zarr metadata")
        
        # once the dataset is loaded, this will become an array of len(dates) where the value corresponds to which node's cache a file is in
        # -1 => filesystem
        self.cache_registry=None
        self._init_cache(self.cache_root)
       
        # One HTTP server per node, run by the node leader only
        self.root_port = 8000
        self.port = self.root_port + self.node_id
        if self.is_node_leader:
            self._start_server(self.cache_root, self.port)
        # Barrier so all local ranks wait for the server to be up
        dist.barrier(self.proc_group)
        
        # Inject cache wrapper into the underlying datasets
        self._inject_cache_wrapper()
        
        LOGGER.info(f"{self.rank=}: Initalised a shared cache under {self.cache_path}")
        self.is_initalised=True

        # Open remote zarrs once, store the open handles keyed by node_id
        #dist.barrier(self.proc_group)
        self.remote_zarrs=[None] * self.num_nodes
        for node in range(self.num_nodes):
            if node != self.node_id:
                LOGGER.info(f"Rank {self.rank}: Opening zarr interface to remote cache of node {node} at {self._get_remote_cache_url(node)}")
                remote_cache_url=self._get_remote_cache_url(node)
                self.remote_zarrs[node] = zarr.open(remote_cache_url, mode='r')
                LOGGER.debug(f"Rank {self.rank}: Opened zarr interface to remote cache of node {node}.")
    
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


    def _copy_zarr_metadata(self, source_filesystem, target_path):
        """Copy zarr metadata from the source filesystem to the target path. This allows us to use zarr interfaces on the cache."""
        #TODO can i replace with zarr.empty_lik()
        # TODO check that this works when copying from s3
        source_path = Path(source_filesystem) / self.primary_dataset_name
        if not source_path.exists():
            raise FileNotFoundError(f"Source zarr path {source_path} does not exist")

        shutil.copy2(f"{source_filesystem}/data/.zarray", target_path / ".zarray")
        #shutil.copy2(f"{self.filesystem}/data/.zattrs", self.cache_path) #TODO check if this exists and copy if so
        shutil.copy2(f"{source_filesystem}/.zgroup", target_path / ".zgroup")
       
    def _init_cache(self, cache_root, delete_existing_cache=True):
        """ Set up the local cache directory and initialize the cache registry (list of cache contents). """
        
        if self.cache_root is None:
            LOGGER.info("Cache root not given, defaulting to $TMPDIR")
            self.cache_root=Path(os.getenv("TMPDIR"))
        
        #create space under cache_root for cache — single shared dir per node
        assert self.cache_root.exists()
        self.cache_path = Path(f"{self.cache_root}/cache")
        
        # Only the node leader creates/cleans the directory; others wait.
        if self.is_node_leader:
            if self.cache_path.exists() and delete_existing_cache:
                LOGGER.info(f"Existing cache found under {self.cache_path}. Deleting because {delete_existing_cache=}")
                shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(exist_ok=True)
            
            # copy zarr metadata from filesystem
            # This is allows us to access the cache via a zarr interface.
            # This includes accessing zarrs remotely via HTTP.
            self._copy_zarr_metadata(self.dataset_path, self.cache_path)

        # Barrier so non-leaders don't use the dir before it exists
        dist.barrier(self.proc_group)
        
        self.cache = zarr.open(self.cache_path, mode="a")  # append mode so all ranks can read+write
        self.cache_full=False # prevents subsequent writing when cache is full

        dates = len(self) + 2 # build in some buffer to avoid out of bounds errors, TODO try improve this
        
        if self.cache_registry is None:
            # Keep cache_registry on CPU with shared memory for DataLoader worker access
            # The shared memory is required otherwise the dataloaders can't see updates to the cache registry
            # Values are node_ids (not ranks); -1 => not cached anywhere
            self.cache_registry = torch.zeros(dates, dtype=torch.int32, device='cpu').share_memory_()
            self.cache_registry[:] = -1
            LOGGER.info(f"Rank {self.rank}: Created cache registry on CPU with shared memory for {dates} dates")
            
    def _start_server(self, directory, port):
        directory = Path(self.cache_root).resolve()
        handler=http.server.SimpleHTTPRequestHandler
        
        def no_logging(*args):
            return
        handler.log_message=no_logging #by default the http server logs a lot to stdout, this silences it
        
        handler = partial(handler, directory=str(directory))
        
        # Allow socket reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        
        # Store server metadata for proper shutdown
        self.httpd = httpd
        self.server_thread = thread
        
        LOGGER.info(f"Serving {directory} on http://{self.hostnames[self.node_id]}:{port}")
        
    def _get_all_hostnames(self, suffix:str=None):
        """Gather the hostname of every rank into a list (length = world_size).

        Appends optional suffix to hostnames if given."""
        my_host = socket.gethostname()

        # Gather all hostnames as Python objects
        hostnames = [None for _ in range(self.world_size)]
        dist.all_gather_object(hostnames, my_host, self.proc_group)
        hostnames = list(dict.fromkeys(hostnames)) #deduplicate to get list of unique hostnames
        assert len(hostnames) == self.num_nodes, f"Expected {self.num_nodes} unique hostnames but got {len(hostnames)}: {hostnames}"
        if suffix is not None:
            #append the ib interface to the end of the hostnames
            hostnames = [hostname + suffix for hostname in hostnames]
        LOGGER.info(f"{self.rank=} {hostnames=}")
        return hostnames
    
    def priority_reduce(self, stacked, cache_registry, node_id):
        """ reduces global cache registries by taking local node first then any other node"""
        valid_mask = stacked != -1
        self_mask = stacked == node_id
        has_self = self_mask.any(dim=0)

        global_cache_registry = torch.full_like(cache_registry, -1)

        # Prefer own node
        global_cache_registry[has_self] = node_id

        # Fallback to any other node
        remaining = ~has_self
        if remaining.any():
            temp = stacked.clone()
            temp[~valid_mask] = -1
            fallback = temp.max(dim=0).values
            global_cache_registry[remaining] = fallback[remaining]

        return global_cache_registry

    def update_global_view(self) -> None:
        """ Communicate with other procs and share who has what files in their caches. Should be called after an epoch."""

        all_gather_buffer = [
            torch.zeros_like(self.cache_registry, device="cpu")
            for _ in range(self.world_size)
        ]

        LOGGER.info(f"Starting all gather (local cache registry: {self.cache_registry})")
        dist.all_gather(all_gather_buffer, self.cache_registry, group=self.proc_group)

        self.cache_registry.copy_(self.priority_reduce(torch.stack(all_gather_buffer), self.cache_registry, self.node_id))
        num_cached = (self.cache_registry != -1).sum().item()
        LOGGER.info(f"Rank {self.rank}: Global cache registry updated. Cache now aware of {num_cached} cached items across all nodes")


    def _get_remote_cache_url(self, remote_node_id):
        remote_host = self.hostnames[remote_node_id]
        port = self.root_port + remote_node_id
        remote_cache_path = "cache" #just need relative path here, will be added to the root dir of our http server (e.g. $TMPDIR)
        return f"http://{remote_host}:{port}/{remote_cache_path}"
        
    def fetch(self, date, verbose=False) -> np.ndarray:
        """ Reads cache regsitry, based on result fetches file from local SSD, remote SSD or filesytem"""

        #LOGGER.info(f"Rank {self.rank}: Fetching date {date} (verbose={verbose})")

        # setup distributed access for the cache on first fetch
        # This has to happen at the beginning of runtime not initalisation
        if not self.is_initalised:
            self.setup()
        
        self.total_fetches += 1

        # cache_location: -1 => cache miss, otherwise gives node_id of the cache hit
        cache_location = self.cache_registry[date].item()
        
        # Get the primary dataset's raw data for caching
        # Use self.ds_train to access our cached property
        primary_dataset = self.ds_train.datasets[self.primary_dataset_name]
        primary_data = primary_dataset.data  # Underlying dataset (e.g., Zarr)
        
        # Check if we're accessing the wrapper itself (avoid infinite recursion)
        if isinstance(primary_data, CachedDataWrapper):
            primary_data = primary_data.original_data

        if cache_location == -1:
            #Cache miss, go to filesystem
            
            self.stats['miss'] += 1
            data = primary_data[date]

            # try to write to cache if there is space
            if not self.cache_full:
                # add data to the shared node cache
                try:
                    self.cache[date] = data
                    self.cache_registry[date] = self.node_id
                #TODO calculate and manage space rather then relying on catching an error
                except OSError as e:
                    if e.errno == errno.ENOSPC: 
                        self.cache_full=True
                        LOGGER.info(f"Rank {self.rank}: Cache full! No more writing")
                    else:
                        raise e
            
        elif self.node_id == cache_location:
            #Cache hit on local node SSD – read from shared zarr cache
            self.stats['local'] += 1
            data = self.cache[date]            
                
        else:
            #cache hit on remote node SSD – fetch via HTTP zarr
            self.stats['remote'] += 1

            # TODO move to end of setup
            # tried before and i got error opening '' , so this could be racey
            #if self.remote_zarrs[cache_location] is None:
            #    remote_cache_url=self._get_remote_cache_url(cache_location)
            #    self.remote_zarrs[cache_location] = zarr.open(remote_cache_url, mode='r')
            #    LOGGER.debug(f"Rank {self.rank}: Opened zarr interface to remote cache of node {cache_location}.")

            data = self.remote_zarrs[cache_location][date]
            
        if self.rank == 0 and (verbose or (self.total_fetches % 10 == 0)):
            LOGGER.info(f"Rank {self.rank}: hits_local={self.stats['local']}, hits_remote={self.stats['remote']}, misses={self.stats['miss']}")
            
        return data

    def __iter__(self) -> np.ndarray:
        dates=len(self)
        for date in range(dates):
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
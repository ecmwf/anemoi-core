import os
import io
import errno
import fcntl
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
import urllib3
from multiprocessing import Value

from anemoi.datasets import open_dataset

import numpy as np
import socket
import threading
import http.server


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
        self.dataset_names = self.ds.dataset_names
        
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
                
        self.global_rank = dist.get_rank(self.proc_group)
        self.rank = self.global_rank % 4 # TODO pass system.hardware.gpus_per_node here
        self.local_rank = self.rank  # alias for clarity
        
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        
        # Set the CUDA device before any collective operations (required for NCCL)
        if torch.cuda.is_available() and self.device.type == "cuda":
            #TODO(cathal) fix silly hack which will only work on single nodes
            torch.cuda.set_device(self.rank)
            LOGGER.info(f"Rank {self.rank}: Set CUDA device to {self.rank}")
        
        self.world_size = dist.get_world_size(self.proc_group)
        self.hostnames = self._get_all_hostnames()
        self.proc_group = dist.new_group(ranks=None, backend="gloo")

        # Build a mapping from hostname -> node_id (integer) and figure out
        # which node this rank lives on.  All ranks on the same node share
        # the same node_id and the same cache directory.
        unique_hosts = sorted(set(self.hostnames))
        self._host_to_node_id = {h: i for i, h in enumerate(unique_hosts)}
        self.node_id = self._host_to_node_id[self.hostnames[self.global_rank]]
        self.num_nodes = len(unique_hosts)
        # For each node pick the lowest global rank on that node – that rank
        # is responsible for creating the cache dir and running the HTTP server.
        self._node_leader_rank = {}
        for grank, host in enumerate(self.hostnames):
            nid = self._host_to_node_id[host]
            if nid not in self._node_leader_rank:
                self._node_leader_rank[nid] = grank
        self.is_node_leader = (self.global_rank == self._node_leader_rank[self.node_id])
        LOGGER.info(f"Rank {self.rank} (global {self.global_rank}): node_id={self.node_id}, is_node_leader={self.is_node_leader}")

        # Determine which dataset(s) we're working with
        # For MultiDataset, cache the first dataset by default
        # Use self.ds_train which uses our cached property
        self.dataset_names = list(self.ds_train.datasets.keys())
        self.primary_dataset_name = self.dataset_names[0]
        LOGGER.info(f"DatasetCache: Found datasets {self.dataset_names}, using '{self.primary_dataset_name}' for caching")
        
        #creates space under self.cache_dir
        self.filesystem= f"{self.dataset_path}" #TODO read from zarr metadata
        
        # once the dataset is loaded, this will become an array of len(dates) where the value corresponds to which node's cache a file is in
        # -1 => filesystem
        self.cache_registry=None
        self._init_cache()
       
        # One HTTP server per node, run by the node leader only
        self.root_port = 8000
        self.port = self.root_port + self.node_id
        if self.is_node_leader:
            self._start_server()
        # Barrier so all local ranks wait for the server to be up
        dist.barrier(self.proc_group)
        
        # Inject cache wrapper into the underlying datasets
        self._inject_cache_wrapper()
        
        LOGGER.info(f"{self.rank=}: Initalised a shared cache under {self.cache_path}")
        self.is_initalised=True

        # Per-node persistent HTTP connection pools for remote cache fetches.
        # Keyed by node_id; created lazily on first remote hit.
        self._http_pools: dict[int, urllib3.HTTPConnectionPool] = {}
    
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
        
        #create space under cache_root for cache — single shared dir per node
        assert self.cache_root.exists()
        self.cache_path = Path(f"{self.cache_root}/cache")
        
        # Only the node leader creates/cleans the directory; others wait.
        if self.is_node_leader:
            if self.cache_path.exists() and delete_existing_cache:
                LOGGER.info(f"Existing cache found under {self.cache_path}. Deleting because {delete_existing_cache=}")
                shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(exist_ok=True)
        # Barrier so non-leaders don't use the dir before it exists
        dist.barrier(self.proc_group)
        
        self.cache_full=False # prevents subsequent writing when cache is full
        # File-based lock so multiple local ranks don't write the same .npy concurrently
        self._write_lock_path = self.cache_path / ".write_lock"
       
        dates = len(self) + 2 # build in some buffer to avoid out of bounds errors, will be cleaned up in the future with a more robust solution than using integers for cache registry
        #dates = self.ds.size
        #dates = self.ds.ds_train.data.shape[0]
        
        if self.cache_registry is None:
            # Keep cache_registry on CPU with shared memory for DataLoader worker access
            # Values are node_ids (not ranks); -1 => not cached anywhere
            self.cache_registry = torch.zeros(dates, dtype=torch.int32, device='cpu').share_memory_()
            self.cache_registry[:] = -1
            LOGGER.info(f"Rank {self.rank}: Created cache registry on CPU with shared memory for {dates} dates")
            
        self.remote_cache_roots=self._get_all_cache_roots()
        
    
    def _start_server(self):
        """Start a threaded HTTP server to serve the cache directory.

        Uses http.server.ThreadingHTTPServer so concurrent requests from
        multiple ranks are handled in parallel threads instead of
        sequentially (the main bottleneck of the old single-threaded
        TCPServer approach).  Combined with serving single .npy files
        (one GET per date) this is comparable to nginx for this workload.
        """
        directory = str(Path(self.cache_root).resolve())

        # Build a handler rooted at cache_root with logging suppressed
        handler_cls = type(
            "_QuietHandler",
            (http.server.SimpleHTTPRequestHandler,),
            {"log_message": lambda *_args, **_kw: None},
        )
        # Python 3.7+ – set the directory the handler serves
        original_init = handler_cls.__init__

        def _patched_init(self_h, *args, **kwargs):
            kwargs["directory"] = directory
            original_init(self_h, *args, **kwargs)

        handler_cls.__init__ = _patched_init

        self.httpd = http.server.ThreadingHTTPServer(("", self.port), handler_cls)
        # Allow fast restart after crashes
        self.httpd.allow_reuse_address = True
        # TCP_NODELAY on the listening socket for lower latency
        self.httpd.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        LOGGER.info(
            f"Rank {self.rank}: Threaded HTTP server serving {directory} "
            f"on http://{self._get_hostname(self.rank)}:{self.port}"
        )
        
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
        """Stop the threaded HTTP server (only called on node leader)."""
        try:
            if hasattr(self, 'httpd') and self.httpd is not None:
                LOGGER.info(f"Rank {self.rank}: Shutting down HTTP server on port {self.port}")
                self.httpd.shutdown()
                self.httpd.server_close()
                LOGGER.info(f"Rank {self.rank}: HTTP server shut down successfully")
        except Exception as e:
            LOGGER.warning(f"Rank {self.rank}: Error shutting down server: {e}")

    def __del__(self):
       if getattr(self, 'is_node_leader', False):
           self._shutdown_server()

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


    def _get_http_pool(self, remote_node_id) -> urllib3.HTTPConnectionPool:
        """Return (or lazily create) a persistent connection pool to a remote node."""
        if remote_node_id not in self._http_pools:
            remote_global_rank = self._node_leader_rank[remote_node_id]
            remote_host = self._get_hostname(remote_global_rank)
            port = self.root_port + remote_node_id
            self._http_pools[remote_node_id] = urllib3.HTTPConnectionPool(
                host=remote_host,
                port=port,
                maxsize=4,       # allow a few concurrent connections
                block=False,
                retries=1,
            )
            LOGGER.info(f"Rank {self.rank}: Opened persistent HTTP pool to node {remote_node_id} ({remote_host}:{port})")
        return self._http_pools[remote_node_id]

    def _get_remote_file_url(self, remote_node_id, date):
        """Build the URL for a single .npy file on a remote node's HTTP server."""
        # Pick any rank on that node to get the hostname
        remote_global_rank = self._node_leader_rank[remote_node_id]
        remote_host = self._get_hostname(remote_global_rank)
        port = self.root_port + remote_node_id
        return f"http://{remote_host}:{port}/cache/{date}.npy"
        
    def check_cache(self, date) -> list[int]:
        """ checks either local or global registry. returns list of node_ids containing file in their SSD"""
        # cache_registry stores node_ids, not ranks
        cached_node = self.cache_registry[date].item()
        if cached_node == -1:
            return []
        return [cached_node]

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
            #Cache miss, go to filesystem
            
            self.cache_misses.value += 1
            data = primary_data[date]
            
            if not self.cache_full:
                # add data to the shared node cache as .npy file
                # use a file lock so multiple local ranks don't write the same date concurrently
                npy_path = self.cache_path / f"{date}.npy"
                if not npy_path.exists():  # quick non-locked check to skip if already written by peer
                    try:
                        lock_fd = open(self._write_lock_path, 'w')
                        fcntl.flock(lock_fd, fcntl.LOCK_EX)
                        try:
                            if not npy_path.exists():  # double-check under lock
                                np.save(npy_path, data)
                                self.cache_registry[date] = self.node_id
                        finally:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                            lock_fd.close()

                    #TODO calculate and manage space rather then relying on catching an error
                    except OSError as e:
                        if e.errno == errno.ENOSPC or "Not enough free space" in str(e):
                            self.cache_full=True
                            # Clean up the partial .npy file so it doesn't get served
                            npy_path.unlink(missing_ok=True)
                            LOGGER.info(f"Rank {self.rank}: Cache full! No more writing")
                        else:
                            raise e
                else:
                    # Another local rank already wrote this date
                    self.cache_registry[date] = self.node_id
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: CACHE MISS on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
            
        elif self.node_id in cache_hits:
            #Cache hit on local node SSD – read .npy straight from disk
            
            self.cache_hits_local.value += 1
            data = np.load(self.cache_path / f"{date}.npy")
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: LOCAL CACHE HIT on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
                
        else:
            #cache hit on remote node SSD – single HTTP GET
            
            self.cache_hits_remote.value += 1
            remote_node_id = cache_hits[0]
            url = self._get_remote_file_url(remote_node_id, date)
            
            try:
                pool = self._get_http_pool(remote_node_id)
                resp = pool.request("GET", f"/cache/{date}.npy", preload_content=True)
                if resp.status == 200:
                    data = np.load(io.BytesIO(resp.data))
                else:
                    raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}")
            except Exception as e:
                LOGGER.warning(f"Rank {self.rank}: Failed to fetch date {date} from node {remote_node_id}: {e}. Falling back to filesystem.")
                data = primary_data[date]
            
            if verbose or (self.total_fetches.value % 10 == 0):
                LOGGER.info(f"Rank {self.rank}: REMOTE CACHE HIT (node {remote_node_id}) on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
            
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
    
    #TODO currently does not get called
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch == 0:
            self.update_global_view()
        

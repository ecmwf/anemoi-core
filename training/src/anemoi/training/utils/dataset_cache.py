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
import struct
import io

from abc import ABC, abstractmethod


import pytorch_lightning as pl
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

import logging
LOGGER = logging.getLogger(__name__)


def _recv_exact(sock, nbytes):
    """Receive exactly *nbytes* from *sock*, or return None on clean EOF."""
    parts = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))  # up to 1 MB per recv
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b''.join(parts)


class TCPCacheServer:
    """TCP server that serves numpy arrays from the local zarr cache.

    Protocol (persistent connection, many requests per connection):
      Request:  4 bytes  – date index (uint32, big-endian)
      Response: 8 bytes  – payload length (uint64, big-endian)
                N bytes  – uncompressed .npy data (``np.save`` format)
    """

    def __init__(self, cache, port, host=""):
        self.cache = cache
        self.port = port
        self.host = host

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(64)
        self._server_socket = srv
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._server_socket.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle(self, conn):
        try:
            while True:
                hdr = _recv_exact(conn, 4)
                if hdr is None:
                    break
                date = struct.unpack("!I", hdr)[0]
                arr = np.asarray(self.cache[date])
                buf = io.BytesIO()
                np.save(buf, arr)
                payload = buf.getvalue()
                conn.sendall(struct.pack("!Q", len(payload)) + payload)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            conn.close()


class TCPCacheClient:
    """Persistent-connection client for :class:`TCPCacheServer`.

    Lazily connects on first ``fetch`` and reconnects automatically on failure.
    Supports ``client[date]`` indexing for drop-in compatibility with zarr arrays.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._sock = None

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self._sock = s

    def fetch(self, date):
        if self._sock is None:
            self._connect()
        try:
            self._sock.sendall(struct.pack("!I", date))
            hdr = _recv_exact(self._sock, 8)
            if hdr is None:
                raise ConnectionError("server closed connection")
            length = struct.unpack("!Q", hdr)[0]
            payload = _recv_exact(self._sock, length)
            if payload is None:
                raise ConnectionError("server closed connection")
            return np.load(io.BytesIO(payload))
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            self._sock = None
            self._connect()
            return self.fetch(date)

    def __getitem__(self, date):
        return self.fetch(date)

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


#TODO change to overwriting the dataset insetad of data module?
class DatasetCache(AnemoiDatasetsDataModule):
    def __init__(self, ds, cache_root, dataset_path, proc_group=None, hostname_suffix=None, log_cache_stats=True):
        self.ds=ds
        self.cache_root = Path(cache_root)
        self.dataset_path=dataset_path
        self.proc_group = proc_group
        
        # For multi-dataset scenarios, we cache the first dataset by default
        # In the future, this could be made configurable
        self.dataset_names = None
        self.primary_dataset_name = None
        
        self.is_initalised = False

        # optional suffix which will be appended to hostnames
        self.hostname_suffix=hostname_suffix
        
        self.log_cache_stats = log_cache_stats
        if self.log_cache_stats:
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

    @abstractmethod
    def _start_server(self, directory, port):
        """Start the server that will serve cached files to other nodes. Only called on node leaders."""
        raise NotImplementedError("_start_server must be implemented by subclasses of DatasetCache")

    @abstractmethod
    def _fetch_remote(self, date, verbose=False):
        """Fetch a single date from a remote nodes cache."""
        pass

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
        self.hostnames = self._get_all_hostnames(suffix=self.hostname_suffix)
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
        LOGGER.info(f"DatasetCache: Found datasets {self.dataset_names}, using '{self.primary_dataset_name}' for zarr metadata")
        
        #creates space under self.cache_dir
        self.filesystem= f"{self.dataset_path}" #TODO read from zarr metadata
        
        # once the dataset is loaded, this will become an array of len(dates) where the value corresponds to which node's cache a file is in
        # -1 => filesystem
        self.cache_registry=None
        self._init_cache()
       
        # One server per node (HTTP for zarr backend, TCP for tcp backend),
        # run by the node leader only.
        self.root_port = 8000
        self.port = self.root_port + self.node_id
        if self.is_node_leader:
            self._start_server(self.cache_root, self.port)
        # Barrier so all local ranks wait for the server to be up
        dist.barrier(self.proc_group)
        
        LOGGER.info(f"{self.rank=}: Initalised a shared cache under {self.cache_path}")
        self.is_initalised=True

        # Remote connection handles – opened lazily in fetch()
        self.remote_zarrs = [None] * self.num_nodes
        self.remote_tcp_clients = [None] * self.num_nodes
    
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
            
            #copy zarr metadata from filesystem
            # This is needed so we can load chunks from the cache like we would from the remote zarr
            # BUT we will have a lot of gaps in the local copy, so we will have a seperate list self.cache_registry of which elements are present or not
            #TODO replace with zarr.empty_like()
            shutil.copy2(f"{self.filesystem}/data/.zarray", self.cache_path)
            #shutil.copy2(f"{self.filesystem}/data/.zattrs", self.cache_path) #some datasets dont have, not sure if its needed tbh
            shutil.copy2(f"{self.filesystem}/.zgroup", self.cache_path)
        # Barrier so non-leaders don't use the dir before it exists
        dist.barrier(self.proc_group)
        
        self.cache = zarr.open(self.cache_path, mode="a")  # append mode so all ranks can read+write
        self.cache_full=False # prevents subsequent writing when cache is full
       
        dates = len(self) + 2 # build in some buffer to avoid out of bounds errors, will be cleaned up in the future with a more robust solution than using integers for cache registry
        
        if self.cache_registry is None:
            # Keep cache_registry on CPU with shared memory for DataLoader worker access
            # Values are node_ids (not ranks); -1 => not cached anywhere
            self.cache_registry = torch.zeros(dates, dtype=torch.int32, device='cpu').share_memory_()
            self.cache_registry[:] = -1
            LOGGER.info(f"Rank {self.rank}: Created cache registry on CPU with shared memory for {dates} dates")
            
        self.remote_cache_roots=self._get_all_cache_roots()
        
    def _get_hostname(self, rank):
        return self.hostnames[rank]
    
    def _get_all_hostnames(self, suffix:str=None):
        """Gather the hostname of every rank into a list (length = world_size).

        Appends optional suffix to hostnames if given."""
        my_host = socket.gethostname()

        # Gather all hostnames as Python objects
        hostnames = [None for _ in range(self.world_size)]
        dist.all_gather_object(hostnames, my_host, self.proc_group)
        if suffix is not None:
            #append the ib interface to the end of the hostnames
            hostnames = [hostname + suffix for hostname in hostnames]
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
    
    @abstractmethod
    def _shutdown_server(self):
        """Shutdown the HTTP server and close the socket."""
        pass

    def __del__(self):
        #loops here and gets stuck
        #if getattr(self, 'is_node_leader', False):
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

    def check_cache(self, date) -> list[int]:
        """ checks either local or global registry. returns list of node_ids containing file in their SSD"""
        # cache_registry stores node_ids, not ranks
        cached_node = self.cache_registry[date].item()
        if cached_node == -1:
            return []
        return [cached_node]

    def _add_to_cache(self, date, data):
        if not self.cache_full:
            # add data to the shared node cache
            try:
                self.cache[date] = data
                self.cache_registry[date] = self.node_id

            #TODO calculate and manage space rather then relying on catching an error
            except OSError as e:
                if e.errno == errno.ENOSPC or "Not enough free space" in str(e):
                    self.cache_full=True
                    LOGGER.info(f"Rank {self.rank}: Cache full! No more writing")
                else:
                    raise e

    def fetch(self, date, verbose=False) -> np.ndarray:
        """ Reads cache regsitry, based on result fetches file from local SSD, remote SSD or filesytem"""

        if self.log_cache_stats:
            self.total_fetches.value += 1

        cache_hits = self.check_cache(date)
        
        if len(cache_hits) == 0:
            #Cache miss, go to filesystem and add to cache
            data = self.ds_train.datasets[self.primary_dataset_name][date]
            self._add_to_cache(date, data)
            
            # Logging and stats update
            if self.log_cache_stats:
                self.cache_misses.value += 1
                if verbose or (self.total_fetches.value % 10 == 0):
                    LOGGER.info(f"Rank {self.rank}: CACHE MISS on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
            
        elif self.node_id in cache_hits:
            #Cache hit on local node SSD – read from shared zarr cache
            data = self.cache[date]            

            # Logging and stats update
            if self.log_cache_stats:
                self.cache_hits_local.value += 1
                if verbose or (self.total_fetches.value % 10 == 0):
                    LOGGER.info(f"Rank {self.rank}: LOCAL CACHE HIT on date {date} (total: hits_local={self.cache_hits_local.value}, hits_remote={self.cache_hits_remote.value}, misses={self.cache_misses.value})")
                
        else:
            #cache hit on remote node SSD
            remote_node_id = cache_hits[0]
            data = self._fetch_remote(date, remote_node_id, verbose=verbose)

            # Logging and stats update
            if self.log_cache_stats:
                self.cache_hits_remote.value += 1
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
        return self.fetch(index)

    def __len__(self):
        # MultiDataset.valid_date_indices contains all valid time indices
        # Use self.ds_train to access our cached property
        return len(self.ds_train.valid_date_indices)
    
class ZarrDatasetCache(DatasetCache):
    """ Should implement _start_server, _fetch_remote and optionally _shutdown_server"""
    def _start_server(self, directory, port):
        #directory = Path("/").resolve()
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
        
        LOGGER.info(f"Serving {directory} on http://{self._get_hostname(self._node_leader_rank[self.node_id])}:{port}")

    def _get_remote_cache_url(self, remote_node_id):
        remote_global_rank = self._node_leader_rank[remote_node_id]
        remote_host = self._get_hostname(remote_global_rank)
        port = self.root_port + remote_node_id
        remote_cache_path = "cache" #just need relative path here, will be added to the root dir of our http server (e.g. $TMPDIR)
        return f"http://{remote_host}:{port}/{remote_cache_path}"


    def _fetch_remote(self, date, remote_node_id, verbose=False):
        if self.remote_zarrs[remote_node_id] is None:
            remote_cache_url=self._get_remote_cache_url(remote_node_id)
            try:
                self.remote_zarrs[remote_node_id] = zarr.open(remote_cache_url, mode='r')
                LOGGER.info(f"Rank {self.rank}: Opened zarr interface to remote cache of node {remote_node_id}.")
            except (PathNotFoundError, KeyError) as e:
                LOGGER.info(f"Error opening remote date {date} from node {remote_node_id} to {self.rank}. full error: {e}. falling back to filesystem.")
                data = primary_data[date]
        data = self.remote_zarrs[remote_node_id][date]


class TCPDatasetCache(DatasetCache):
    """ Should implement _start_server, _fetch_remote and optionally _shutdown_server"""
    def _start_server(self, directory, port):
        """Start a TCP cache server on this node leader."""
        self.tcp_server = TCPCacheServer(self.cache, port)
        self.tcp_server.start()
        LOGGER.info(f"Serving cache via TCP on {self._get_hostname(self._node_leader_rank[self.node_id])}:{port}")

    def _shutdown_server(self):
        try:
            if hasattr(self, 'httpd') and self.httpd is not None:
                LOGGER.info(f"Rank {self.rank}: Shutting down HTTP server on port {self.port}")
                #self.httpd.shutdown()
                #TODO doesnt kill server, use pkill instead
                #if hasattr(self, 'server_thread') and self.server_thread is not None:
                #    self.server_thread.join(timeout=5)
                # Close the socket to free the port
                #self.httpd.server_close()
                LOGGER.info(f"Rank {self.rank}: HTTP server shut down successfully")
        except Exception as e:
            LOGGER.warning(f"Rank {self.rank}: Error shutting down server: {e}")

    def _get_remote_tcp_address(self, remote_node_id):
        """Return (hostname, port) for the TCP cache server on *remote_node_id*."""
        remote_global_rank = self._node_leader_rank[remote_node_id]
        remote_host = self._get_hostname(remote_global_rank)
        port = self.root_port + remote_node_id
        return remote_host, port
        

    def _fetch_remote(self, date, remote_node_id, verbose=False):
        if self.remote_tcp_clients[remote_node_id] is None:
            host, port = self._get_remote_tcp_address(remote_node_id)
            self.remote_tcp_clients[remote_node_id] = TCPCacheClient(host, port)
            LOGGER.info(f"Rank {self.rank}: Opened TCP connection to remote cache of node {remote_node_id} ({host}:{port}).")
        data = self.remote_tcp_clients[remote_node_id].fetch(date)
    

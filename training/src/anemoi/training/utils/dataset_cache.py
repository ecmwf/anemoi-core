import errno
import fcntl
import http.server
import io
import logging
import os
import shutil
import socket
import socketserver
import struct
import threading
from functools import partial
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import zarr
from zarr.convenience import PathNotFoundError

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.multidataset import MultiDataset

LOGGER = logging.getLogger(__name__)


class CachedMultiDataset(MultiDataset):
    """``MultiDataset`` whose primary dataset is served from the distributed cache.

    This is the seam that routes data reads through the cache. It overrides only
    :meth:`get_sample`: for the *primary* dataset the per-date arrays are pulled
    from the :class:`DatasetCache` (local SSD, remote SSD or filesystem), while
    every other dataset is read normally.

    An existing ``MultiDataset`` instance is upgraded in place by re-assigning its
    ``__class__`` (see :meth:`DatasetCache.train_dataloader`). This keeps all the
    instance state (open datasets, shard/comm-group info set by the strategy,
    worker partitioning) and the inherited ``__iter__`` / ``per_worker_init`` /
    ``set_comm_group_info`` logic, so the cache plugs in without monkeypatching
    the underlying ``.data`` reader or re-implementing the iteration protocol.
    """

    # Set when an instance is upgraded; see DatasetCache.train_dataloader.
    _cache: "DatasetCache"

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        start = index + self.data_relative_date_indices[0]
        end = index + self.data_relative_date_indices[-1] + 1
        if len(self.data_relative_date_indices) > 1:
            timeincrement = self.data_relative_date_indices[1] - self.data_relative_date_indices[0]
        else:
            timeincrement = 1  # single time step
        time_indices = slice(start, end, timeincrement)

        primary_name = self._cache.primary_dataset_name

        x = {}
        for name, dataset in self.datasets.items():
            if self.shard_shapes is not None and self.shard_shapes[name] is not None:
                shard_start, shard_end = get_partition_range(self.shard_shapes[name], self.reader_group_rank)
                grid_indices = slice(shard_start, shard_end)
            else:
                grid_indices = slice(None)

            if name == primary_name:
                x[name] = self._cache.get_primary_sample(time_indices, grid_indices)
            else:
                x[name] = dataset.get_sample(time_indices, grid_indices)

        return x


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
    """Receive exactly *nbytes* from *sock*, or return None on clean EOF."""
    parts = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))  # up to 1 MB per recv
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    result = b"".join(parts)
    if result is None:
        msg = "Socket connection closed while receiving data"
        raise ConnectionError(msg)
    return result


class TCPCacheServer:
    """TCP server that serves numpy arrays from the local zarr cache.

    Protocol (persistent connection, many requests per connection):
      Request:  4 bytes  : date index (uint32, big-endian)
      Response: 8 bytes  : payload length (uint64, big-endian)
                N bytes  : uncompressed .npy data (``np.save`` format)
    """

    def __init__(self, cache: MultiDataset, port: int, host: str = ""):
        self.cache = cache
        self.port = port
        self.host = host

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(64)
        self._server_socket = srv
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._server_socket.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle(self, conn: socket.socket) -> None:
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

    def shutdown(self) -> None:
        self._server_socket.close()
        self._thread.join(timeout=5)


class TCPCacheClient:
    """Persistent-connection client for :class:`TCPCacheServer`.

    Lazily connects on first ``fetch`` and reconnects automatically on failure.
    Supports ``client[date]`` indexing for drop-in compatibility with zarr arrays.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._sock = None

    def _connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self._sock = s

    def fetch(self, date: int) -> np.ndarray:
        if self._sock is None:
            self._connect()
        try:
            self._sock.sendall(struct.pack("!I", date))
            hdr = _recv_exact(self._sock, 8)
            length = struct.unpack("!Q", hdr)[0]
            payload = _recv_exact(self._sock, length)
            return np.load(io.BytesIO(payload))
        except (ConnectionResetError, BrokenPipeError, ConnectionError) as e:
            msg = f"Connection to cache server at {self.host}:{self.port} lost, retrying"
            raise ConnectionError(msg) from e

    def __getitem__(self, date: int) -> np.ndarray:
        return self.fetch(date)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class DatasetCache(pl.LightningDataModule):
    def __init__(
        self,
        ds: MultiDataset,
        cache_root: Path,
        dataset_path: Path,
        hostname_suffix: str | None = None,
        log_cache_stats: bool | None = True,
        local_only: bool = False,
    ) -> None:
        # Set the wrapped datamodule first so __getattr__ delegation works safely
        # during super().__init__().
        self.ds = ds
        super().__init__()
        self.cache_root = Path(cache_root)
        self.dataset_path = dataset_path

        # For multi-dataset scenarios, we cache the first dataset by default
        # In the future, this could be made configurable
        self.dataset_names = None
        self.primary_dataset_name = None

        self.is_initalised = False

        # When True, remote cache hits are treated as misses: data is always read
        # from the filesystem and written only to the local node SSD. Useful when
        # inter-node bandwidth is scarce or there is only a single node.
        self.local_only = local_only

        # optional suffix which will be appended to hostnames
        self.hostname_suffix = hostname_suffix

        self.log_cache_stats = log_cache_stats
        if self.log_cache_stats:
            # Cache statistics - use shared memory for cross-process visibility
            self.cache_hits_local = Value("i", 0)  # 'i' = signed int
            self.cache_hits_remote = Value("i", 0)
            self.cache_misses = Value("i", 0)
            self.total_fetches = Value("i", 0)

        # Store the wrapped dataset to prevent it from being recreated
        self._cached_ds_train = None
        self.dataset_names = self.ds.dataset_names

        # Initialize cache-related attributes that will be set later in setup()
        # This ensures they exist on the wrapper instance and don't trigger __getattr__
        self.cache_registry = None
        self.remote_zarrs = None
        self.remote_tcp_clients = None

        # Initialize setup-related attributes
        self.global_rank = None
        self.rank = None
        self.local_rank = None
        self.device = None
        self.world_size = None
        self.hostnames = None
        self.proc_group = None
        self._host_to_node_id = None
        self.node_id = None
        self.num_nodes = None
        self._node_leader_rank = None
        self.is_node_leader = None
        self.filesystem = None
        self.root_port = None
        self.port = None

        # Initialize _init_cache-related attributes
        self.cache_path = None
        self.cache = None
        self.cache_full = None
        self.remote_cache_roots = None

    @property
    def ds_train(self) -> MultiDataset:
        """Return the training dataset with cache wrapper applied."""
        if self._cached_ds_train is None:
            self._cached_ds_train = self.ds.ds_train
        return self._cached_ds_train

    # --- pl.LightningDataModule hooks ---------------------------------------
    # These methods are defined on pl.LightningDataModule, so __getattr__ is not
    # triggered for them. Forward them explicitly to the wrapped datamodule.
    def train_dataloader(self):  # noqa: ANN201
        # Make sure the distributed cache is initialised (collective ops). This is
        # called symmetrically on every rank, so it is a safe place to run setup if
        # the on_train_start callback has not already done so.
        if not self.is_initalised:
            self.cache_setup()

        # Upgrade the training MultiDataset in place so its primary-dataset reads are
        # served from the cache. We mutate the existing instance (rather than building
        # a new one) so the strategy's set_comm_group_info / sharding still applies to
        # the dataset the DataLoader actually iterates.
        ds_train = self.ds.ds_train
        if not isinstance(ds_train, CachedMultiDataset):
            ds_train.__class__ = CachedMultiDataset
            ds_train._cache = self

        return self.ds.train_dataloader()

    def val_dataloader(self):  # noqa: ANN201
        return self.ds.val_dataloader()

    def test_dataloader(self):  # noqa: ANN201
        return self.ds.test_dataloader()

    def __getattr__(self, name: str):
        """Called *only* if the attribute wasn't found on self.

        Delegate to the underlying dataset (self.ds).
        """
        # Guard against recursion if ``ds`` itself is not set yet.
        if name == "ds":
            raise AttributeError(name)
        return getattr(self.ds, name)

    def _start_server(self) -> None:
        """Start the server that will serve cached files to other nodes. Only called on node leaders."""
        msg = "_start_server must be implemented by subclasses of DatasetCache"
        raise NotImplementedError(msg)

    def _fetch_remote(self) -> np.ndarray:
        """Fetch a single date from a remote nodes cache."""
        msg = "_fetch_remote must be implemented by subclasses of DatasetCache"
        raise NotImplementedError(msg)

    # split between init and cache_setup to match distributed strategy from PL structure
    # (we need the process group to be initalised)
    # NOTE: named cache_setup (not setup) to avoid clashing with
    # pl.LightningDataModule.setup(stage), which PyTorch Lightning calls during fit.
    def cache_setup(self) -> None:

        # use default global proc group if proc_group is none
        # TODO(cathal): you should be able to use it if you only run on a single gpu
        if not dist.is_initialized():  # TODO(cathal): and check proc_group is valid
            error_msg = "Torch distributed is not initalised, can't use distributed cache"
            raise ValueError(error_msg)

        self.global_rank = dist.get_rank()  # use global proc group
        self.rank = self.global_rank % 4  # TODO(cathal): pass system.hardware.gpus_per_node here
        self.local_rank = self.rank  # alias for clarity

        self.device = torch.device("cuda:" + str(self.rank) if torch.cuda.is_available() else "cpu")

        # Set the CUDA device before any collective operations (required for NCCL)
        if torch.cuda.is_available() and self.device.type == "cuda":
            # TODO(cathal): fix silly hack which will only work on single nodes
            torch.cuda.set_device(self.rank)
            LOGGER.info("Rank %s: Set CUDA device to %s", self.rank, self.rank)

        self.world_size = dist.get_world_size()
        self.hostnames = self._get_all_hostnames(suffix=self.hostname_suffix)
        self.proc_group = dist.new_group(ranks=None, backend="gloo")

        # Build a mapping from hostname -> node_id (integer) and figure out
        # which node this rank lives on.  All ranks on the same node share
        # the same node_id and the same cache directory.
        unique_hosts = sorted(set(self.hostnames))
        self._host_to_node_id = {h: i for i, h in enumerate(unique_hosts)}
        self.node_id = self._host_to_node_id[self.hostnames[self.global_rank]]
        self.num_nodes = len(unique_hosts)
        # For each node pick the lowest global rank on that node,
        # that rank is responsible for creating the cache dir and running the HTTP server.
        self._node_leader_rank = {}
        for grank, host in enumerate(self.hostnames):
            nid = self._host_to_node_id[host]
            if nid not in self._node_leader_rank:
                self._node_leader_rank[nid] = grank
        self.is_node_leader = self.global_rank == self._node_leader_rank[self.node_id]
        LOGGER.info(
            "Rank %s (global %s): node_id=%s, is_node_leader=%s",
            self.rank,
            self.global_rank,
            self.node_id,
            self.is_node_leader,
        )

        # Determine which dataset(s) we're working with
        # For MultiDataset, cache the first dataset by default
        # Use self.ds_train which uses our cached property
        self.dataset_names = list(self.ds_train.datasets.keys())
        self.primary_dataset_name = self.dataset_names[0]
        LOGGER.info(
            "Rank %s: DatasetCache: Found datasets %s, using '%s' for zarr metadata",
            self.rank,
            self.dataset_names,
            self.primary_dataset_name,
        )

        # creates space under self.cache_dir
        self.filesystem = f"{self.dataset_path}"  # TODO(cathal): read from zarr metadata

        # once the dataset is loaded, this will become an array of len(dates)
        # where the value corresponds to which node's cache a file is in
        # -1 => filesystem
        self.cache_registry = None
        self._init_cache()

        # One server per node (HTTP for zarr backend, TCP for tcp backend),
        # run by the node leader only.
        self.root_port = 8000
        self.port = self.root_port + self.node_id
        if self.is_node_leader:
            self._start_server(self.cache_root, self.port)
        # Barrier so all local ranks wait for the server to be up
        dist.barrier(self.proc_group)

        LOGGER.info("Rank %s: Initalised a shared cache under %s", self.rank, self.cache_path)
        self.is_initalised = True

        # Remote connection handles, opened lazily in fetch()
        self.remote_zarrs = [None] * self.num_nodes
        self.remote_tcp_clients = [None] * self.num_nodes

    def _init_cache(self, delete_existing_cache: bool = True) -> None:

        if self.cache_root is None:
            LOGGER.info("Cache root not given, defaulting to $TMPDIR")
            self.cache_root = Path(os.getenv("TMPDIR"))

        # create space under cache_root for cache, single shared dir per node
        assert self.cache_root.exists()
        self.cache_path = Path(f"{self.cache_root}/cache")

        # Only the node leader creates/cleans the directory; others wait.
        if self.is_node_leader:
            if self.cache_path.exists() and delete_existing_cache:
                LOGGER.info(
                    "Rank %s: Existing cache found under %s. Deleting because delete_existing_cache=%s",
                    self.rank,
                    self.cache_path,
                    delete_existing_cache,
                )
                shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(exist_ok=True)

            # copy zarr metadata from filesystem
            # This is needed so we can load chunks from the cache like we would from the remote zarr
            # BUT we will have a lot of gaps in the local copy,
            # so we will have a seperate list self.cache_registry of which elements are present or not
            # TODO(cathal): replace with zarr.empty_like()
            shutil.copy2(f"{self.filesystem}/data/.zarray", self.cache_path)
            shutil.copy2(f"{self.filesystem}/.zgroup", self.cache_path)
        # Barrier so non-leaders don't use the dir before it exists
        dist.barrier(self.proc_group)

        self.cache = zarr.open(self.cache_path, mode="a")  # append mode so all ranks can read+write
        self.cache_full = False  # prevents subsequent writing when cache is full

        dates = len(self) + 2  # build in some buffer to avoid out of bounds errors,
        # will be cleaned up in the future with a more robust solution than using integers for cache registry

        if self.cache_registry is None:
            # Keep cache_registry on CPU with shared memory for DataLoader worker access
            # Values are node_ids (not ranks); -1 => not cached anywhere
            self.cache_registry = torch.zeros(dates, dtype=torch.int32, device="cpu").share_memory_()
            self.cache_registry[:] = -1
            LOGGER.info("Rank %s: Created cache registry on CPU with shared memory for %s dates", self.rank, dates)

        self.remote_cache_roots = self._get_all_cache_roots()

    def _get_hostname(self, rank: int) -> str:
        return self.hostnames[rank]

    def _get_all_hostnames(self, suffix: str | None = None) -> list[str]:
        """Gather the hostname of every rank into a list (length = world_size).

        Appends optional suffix to hostnames if given.
        """
        my_host = socket.gethostname()

        # Gather all hostnames as Python objects
        hostnames = [None for _ in range(self.world_size)]
        dist.all_gather_object(hostnames, my_host, self.proc_group)
        if suffix is not None:
            # append the ib interface to the end of the hostnames
            hostnames = [hostname + suffix for hostname in hostnames]
        LOGGER.info("Rank %s: Hostnames gathered: %s", self.rank, hostnames)
        return hostnames

    def _get_all_cache_roots(self) -> list[str]:
        """Gather the cache roots of every rank into a list (length = world_size).

        needed bc $TMPDIR is different paths on different ranks
        """
        root = self.cache_root

        # Gather all hostnames as Python objects
        roots = [None for _ in range(self.world_size)]
        dist.all_gather_object(roots, root, self.proc_group)
        return roots

    def _shutdown_server(self) -> None:
        """Can be implemented by subclasses if any cleanup is needed."""

    def __del__(self) -> None:
        self._shutdown_server()

    def priority_reduce(self, stacked: torch.Tensor, cache_registry: torch.Tensor, node_id: int) -> torch.Tensor:
        """Reduces global cache registries by taking local node first then any other node."""
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
        """Communicate with other procs and share who has what files in their caches.

        Should be called after an epoch.
        """
        all_gather_buffer = [torch.zeros_like(self.cache_registry, device="cpu") for _ in range(self.world_size)]

        LOGGER.info("Rank %s: Starting all gather (local cache registry: %s)", self.rank, self.cache_registry)
        dist.all_gather(all_gather_buffer, self.cache_registry, group=self.proc_group)

        self.cache_registry.copy_(
            self.priority_reduce(torch.stack(all_gather_buffer), self.cache_registry, self.node_id),
        )
        num_cached = (self.cache_registry != -1).sum().item()
        LOGGER.info(
            "Rank %s: Global cache registry updated. Cache now aware of %s cached items across all nodes",
            self.rank,
            num_cached,
        )

    def check_cache(self, date: int) -> list[int]:
        """Checks either local or global registry. returns list of node_ids containing file in their SSD."""
        # cache_registry stores node_ids, not ranks
        cached_node = self.cache_registry[date].item()
        if cached_node == -1:
            return []
        return [cached_node]

    def _add_to_cache(self, date: int, data: np.ndarray, locking: bool = True) -> None:
        if not self.cache_full:
            try:

                # add to cache
                if locking:
                    lock_path = Path(self.cache_path / f"{date}.lock")
                    with lock_path.open("w") as lock_file:
                        # blocks until lock acquired
                        fcntl.flock(lock_file, fcntl.LOCK_EX)

                        # add data to the shared node cache
                        try:
                            self.cache[date] = data
                            self.cache_registry[date] = self.node_id
                        finally:
                            fcntl.flock(lock_file, fcntl.LOCK_UN)
                else:
                    self.cache[date] = data
                    self.cache_registry[date] = self.node_id

            # TODO(cathal): calculate and manage space rather then relying on catching an error
            except OSError as e:
                if e.errno == errno.ENOSPC or "Not enough free space" in str(e):
                    self.cache_full = True
                    LOGGER.info("Rank %s: Cache full! No more writing", self.rank)
                else:
                    msg = f"Error adding date {date} to cache on rank {self.rank}: {e}"
                    raise OSError(msg) from e

    # TODO(cathal): break into multiple methods
    def fetch(self, date: int, verbose: bool = False) -> np.ndarray:  # noqa: C901
        """Reads cache registry, based on result fetches file from local SSD, remote SSD or filesystem."""
        if self.log_cache_stats:
            self.total_fetches.value += 1

        cache_hits = self.check_cache(date)

        if len(cache_hits) == 0:
            # Cache miss, go to filesystem and add to cache
            data = self.ds_train.datasets[self.primary_dataset_name].data[date]
            self._add_to_cache(date, data)

            # Logging and stats update
            if self.log_cache_stats:
                self.cache_misses.value += 1
                if verbose or (self.total_fetches.value % 10 == 0):
                    LOGGER.info(
                        "Rank %s: CACHE MISS on date %s (total: hits_local=%s, hits_remote=%s, misses=%s)",
                        self.rank,
                        date,
                        self.cache_hits_local.value,
                        self.cache_hits_remote.value,
                        self.cache_misses.value,
                    )

        elif self.node_id in cache_hits:
            # Cache hit on local node SSD, read from shared zarr cache
            data = self.cache[date]

            # Logging and stats update
            if self.log_cache_stats:
                self.cache_hits_local.value += 1
                if verbose or (self.total_fetches.value % 10 == 0):
                    LOGGER.info(
                        "Rank %s: LOCAL CACHE HIT on date %s (total: hits_local=%s, hits_remote=%s, misses=%s)",
                        self.rank,
                        date,
                        self.cache_hits_local.value,
                        self.cache_hits_remote.value,
                        self.cache_misses.value,
                    )

        else:
            if self.local_only:
                # Remote cache hits are disabled: treat as a miss and cache locally.
                data = self.ds_train.datasets[self.primary_dataset_name].data[date]
                self._add_to_cache(date, data)

                if self.log_cache_stats:
                    self.cache_misses.value += 1
                    if verbose or (self.total_fetches.value % 10 == 0):
                        LOGGER.info(
                            "Rank %s: LOCAL-ONLY MISS (remote skipped) on date %s "
                            "(total: hits_local=%s, hits_remote=%s, misses=%s)",
                            self.rank,
                            date,
                            self.cache_hits_local.value,
                            self.cache_hits_remote.value,
                            self.cache_misses.value,
                        )
            else:
                # cache hit on remote node SSD
                remote_node_id = cache_hits[0]
                data = self._fetch_remote(date, remote_node_id)

                # Logging and stats update
                if self.log_cache_stats:
                    self.cache_hits_remote.value += 1
                    if verbose or (self.total_fetches.value % 10 == 0):
                        LOGGER.info(
                            "Rank %s: REMOTE CACHE HIT on date %s (total: hits_local=%s, hits_remote=%s, misses=%s)",
                            self.rank,
                            date,
                            self.cache_hits_local.value,
                            self.cache_hits_remote.value,
                            self.cache_misses.value,
                        )

        return data

    def get_primary_sample(self, time_indices: slice, grid_shard_indices: slice) -> torch.Tensor:
        """Return one training sample for the primary dataset, sourced from the cache.

        Mirrors ``BaseAnemoiReader.get_sample`` (the normal read path) but assembles
        the requested dates from per-date cached arrays via :meth:`fetch` instead of
        reading the underlying zarr directly. Used by :class:`CachedMultiDataset`.
        """
        dates = range(int(time_indices.start), int(time_indices.stop), int(time_indices.step or 1))
        # Each fetch returns one date with shape (variables, ensemble, gridpoints).
        stacked = np.stack([np.asarray(self.fetch(date)) for date in dates], axis=0)
        # (dates, variables, ensemble, gridpoints) -> select grid shard
        stacked = stacked[:, :, :, grid_shard_indices]
        # match BaseAnemoiReader.get_sample layout: dates ensemble gridpoints variables
        stacked = np.transpose(stacked, (0, 2, 3, 1))
        return torch.from_numpy(np.ascontiguousarray(stacked))

    def __len__(self) -> int:
        # MultiDataset.valid_date_indices contains all valid time indices
        # Use self.ds_train to access our cached property
        return len(self.ds_train.valid_date_indices)


class ZarrDatasetCache(DatasetCache):
    """Should implement _start_server, _fetch_remote and optionally _shutdown_server."""

    def _start_server(self, directory: Path, port: int) -> None:
        directory = Path(directory).resolve()
        handler = http.server.SimpleHTTPRequestHandler

        def no_logging(*args) -> None:  # noqa: ARG001
            return

        handler.log_message = no_logging  # by default the http server logs a lot to stdout, this silences it

        handler = partial(handler, directory=str(directory))

        # Allow socket reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        # Store server metadata for proper shutdown
        self.httpd = httpd
        self.server_thread = thread

        LOGGER.info(
            "Serving %s on http://%s:%s",
            directory,
            self._get_hostname(self._node_leader_rank[self.node_id]),
            port,
        )

    def _get_remote_cache_url(self, remote_node_id: int) -> str:
        remote_global_rank = self._node_leader_rank[remote_node_id]
        remote_host = self._get_hostname(remote_global_rank)
        port = self.root_port + remote_node_id
        remote_cache_path = (
            "cache"  # just need relative path here, will be added to the root dir of our http server (e.g. $TMPDIR)
        )
        return f"http://{remote_host}:{port}/{remote_cache_path}"

    def _fetch_remote(self, date: int, remote_node_id: int) -> np.ndarray:

        # Open remote zarr interface if not already done
        if self.remote_zarrs[remote_node_id] is None:
            remote_cache_url = self._get_remote_cache_url(remote_node_id)
            try:
                self.remote_zarrs[remote_node_id] = zarr.open(remote_cache_url, mode="r")
                LOGGER.info("Rank %s: Opened zarr interface to remote cache of node %s.", self.rank, remote_node_id)
            except (PathNotFoundError, KeyError) as e:
                LOGGER.info(
                    "Error opening remote date %s from node %s to %s. full error: %s. falling back to filesystem.",
                    date,
                    remote_node_id,
                    self.rank,
                    e,
                )
                data = self.ds_train.datasets[self.primary_dataset_name].data[date]

        # try to access the data from the remote zarr
        try:
            data = self.remote_zarrs[remote_node_id][date]
        except (KeyError, PathNotFoundError) as e:
            LOGGER.info(
                "Error fetching remote date %s from node %s to %s. full error: %s. falling back to filesystem.",
                date,
                remote_node_id,
                self.rank,
                e,
            )
            data = self.ds_train.datasets[self.primary_dataset_name].data[date]

        return data


class TCPDatasetCache(DatasetCache):
    """Should implement _start_server, _fetch_remote and optionally _shutdown_server."""

    def _start_server(self, directory: Path, port: int) -> None:  # noqa: ARG002
        """Start a TCP cache server on this node leader."""
        # Serve dates straight from this node's local zarr cache.
        self.tcp_server = TCPCacheServer(self.cache, port)
        self.tcp_server.start()
        LOGGER.info("Serving cache via TCP on %s:%s", self._get_hostname(self._node_leader_rank[self.node_id]), port)

    def _shutdown_server(self) -> None:
        if hasattr(self, "tcp_server"):
            try:
                self.tcp_server.shutdown()
                LOGGER.info("Rank %s: TCP server shut down successfully", self.rank)
            except RuntimeError as e:
                LOGGER.warning("Rank %s: Error shutting down server: %s", self.rank, e)

    def _get_remote_tcp_address(self, remote_node_id: int) -> tuple[str, int]:
        """Return (hostname, port) for the TCP cache server on *remote_node_id*."""
        remote_global_rank = self._node_leader_rank[remote_node_id]
        remote_host = self._get_hostname(remote_global_rank)
        port = self.root_port + remote_node_id
        return remote_host, port

    def _fetch_remote(self, date: int, remote_node_id: int) -> np.ndarray:
        if self.remote_tcp_clients[remote_node_id] is None:
            host, port = self._get_remote_tcp_address(remote_node_id)
            self.remote_tcp_clients[remote_node_id] = TCPCacheClient(host, port)
            LOGGER.info(
                "Rank %s: Opened TCP connection to remote cache of node %s (%s:%s).",
                self.rank,
                remote_node_id,
                host,
                port,
            )
        # try to access the data from the remote server
        try:
            data = self.remote_tcp_clients[remote_node_id].fetch(date)
        except (KeyError, PathNotFoundError) as e:
            LOGGER.info(
                "Error fetching remote date %s from node %s to %s. full error: %s. falling back to filesystem.",
                date,
                remote_node_id,
                self.rank,
                e,
            )
            data = self.ds_train.datasets[self.primary_dataset_name].data[date]

        return data

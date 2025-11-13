import os
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
import http.server
import socketserver
from functools import partial

from anemoi.datasets import open_dataset

import zarr
from zarr.convenience import PathNotFoundError
from pathlib import Path
import threading

import numpy as np
import socket
import subprocess


import pytorch_lightning as pl
from anemoi.training.data.datamodule.singledatamodule import AnemoiDatasetsDataModule

import logging
LOGGER = logging.getLogger(__name__)

#TODO change to overwriting the dataset insetad of data module?
class DatasetCache(AnemoiDatasetsDataModule):
    def __init__(self, ds, cache_root, dataset_path, proc_group=None):
        self.ds=ds
        self.cache_root = Path(cache_root)
        self.dataset_path=dataset_path
        self.proc_group = proc_group
        
        self.is_initalised = False
        
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
        self.local_rank = int(os.environ.get("SLURM_LOCALID", -1)) 
        
        self.device = torch.device(f"cuda:{self.local_rank if self.local_rank != -1 else self.rank}" if torch.cuda.is_available() else "cpu")
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
       
        self.root_port = 8000
        self.port = self.root_port + self.rank 
        self._start_server(self.cache_root, self.port) #need to use cache root so that other local caches can be found
        
        LOGGER.info(f"{self.rank=}: Initalised a cache under {self.cache_path}")
        self.is_initalised=True
       
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
        
        httpd = socketserver.TCPServer(("", port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        
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
        if self.global_cache_registry is not None:
            cache_subset=self.global_cache_registry[:,date] # (num_proc) array, -1 if not in cache, non-negative if in cache
        else:
            assert self.cache_registry is not None
            cache_subset = [self.cache_registry[date]]
        cache_hits = [x for x in cache_subset if x != -1]
        return cache_hits

    def fetch(self, date, verbose=True) -> np.ndarray:
        """ Reads cache regsitry, based on result fetches file from local SSD, remote SSD or filesytem"""
        
        cache_hits = self.check_cache(date)
        
        if len(cache_hits) == 0:
            #Cache miss, go to filesystem
            
            data = self.ds[date]
            
            # add data to local cache
            #TODO can i do this async while i return data
            self.cache[date] = data
            self.cache_registry[date] = self.rank
            
            if verbose:
                LOGGER.info(f"{self.rank=}: cache miss on date {date}")
            
        elif self.rank in cache_hits:
            #Cache hit on local SSD
            
            data = self.cache[date]            
            
            if verbose:
                LOGGER.info(f"{self.rank=}: local cache hit on date {date}")
                
        else:
            #cache hit on remote SSD
            
            #take the first cache hit
            remote_rank = cache_hits[0] # TODO could be smarter by picking local ranks 
            remote_cache_url=self._get_remote_cache_url(remote_rank)
            LOGGER.info(f"{self.rank=} accessing {date=} on rank {remote_rank} at {remote_cache_url=}")
            
            try:
                data = zarr.open(remote_cache_url, mode="r")[date]
                #TODO figure out why i get cache misses
            except (PathNotFoundError, KeyError) as e:
                LOGGER.info(f"Error loading remote date {date} from {remote_rank} to {self.rank}. full error: {e}. falling back to filesystem.")
                data = ds[date]
            
            if verbose:
                LOGGER.info(f"{self.rank=}: remote cache hit (rank {remote_rank}) on date {date}")
            
        return data

    def __iter__(self) -> np.ndarray:
        if not self.is_initalised:
            self.setup()
        
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
        #return len(self.ds)
        #TODO Horrible fix this
        return  self.ds.ds_train.data.shape[0]
    
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
    ds = open_dataset(dataset_path)
    cached_ds = DatasetCache(ds, dataset_path)
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

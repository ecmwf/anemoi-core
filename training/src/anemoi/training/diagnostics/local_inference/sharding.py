import os
import sys
import torch
import numpy as np
import xarray as xr

from icecream import ic

import torch.distributed as dist
import torch.multiprocessing as mp  # For launching processes
import datetime
import socket
import subprocess

import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_parallel_info():
    """Reads Slurm env vars, if they exist, to determine if inference is running in parallel"""
    local_rank = int(
        os.environ.get("SLURM_LOCALID", 0)
    )  # Rank within a node, between 0 and num_gpus
    global_rank = int(os.environ.get("SLURM_PROCID", 0))  # Rank within all nodes
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of processes

    return global_rank, local_rank, world_size


def __get_parallel_info():
    """Reads Slurm env vars, if they exist, to determine if inference is running in parallel"""
    local_rank = int(
        os.environ.get("SLURM_LOCALID", 0)
    )  # Rank within a node, between 0 and num_gpus
    global_rank = int(os.environ.get("SLURM_PROCID", 0))  # Rank within all nodes
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of processes

    return global_rank, local_rank, world_size


def init_parallel(device, global_rank, world_size):
    """Creates a model communication group to be used for parallel inference"""

    if world_size > 1:

        master_addr, master_port = init_network()

        # use 'startswith' instead of '==' in case device is 'cuda:0'
        backend = "nccl"
        # if device.startswith("cuda"):
        #    backend = "nccl"
        # else:
        #    backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=datetime.timedelta(minutes=3),
            world_size=world_size,
            rank=global_rank,
        )
        logging.info(
            f"Creating a model comm group with {world_size} devices with the {backend} backend"
        )

        model_comm_group_ranks = np.arange(world_size, dtype=int)
        model_comm_group = torch.distributed.new_group(model_comm_group_ranks)
    else:
        model_comm_group = None

    return model_comm_group


def init_network():
    """Reads Slurm environment to set master address and port for parallel communication"""

    # Get the master address from the SLURM_NODELIST environment variable
    slurm_nodelist = os.environ.get("SLURM_NODELIST")
    if not slurm_nodelist:
        raise ValueError("SLURM_NODELIST environment variable is not set.")

    # Check if MASTER_ADDR is given, otherwise try set it using 'scontrol'
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None:
        LOG.debug("'MASTER_ADDR' environment variable not set. Trying to set via SLURM")
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostname", slurm_nodelist],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            LOG.error(
                "Python could not execute 'scontrol show hostname $SLURM_NODELIST' while calculating MASTER_ADDR. You could avoid this error by setting the MASTER_ADDR env var manually."
            )
            raise err

        master_addr = result.stdout.splitlines()[0]

        # Resolve the master address using nslookup
        try:
            master_addr = socket.gethostbyname(master_addr)
        except socket.gaierror:
            raise ValueError(f"Could not resolve hostname: {master_addr}")

    # Check if MASTER_PORT is given, otherwise generate one based on SLURM_JOBID
    master_port = os.environ.get("MASTER_PORT")
    if master_port is None:
        LOG.debug("'MASTER_PORT' environment variable not set. Trying to set via SLURM")
        slurm_jobid = os.environ.get("SLURM_JOBID")
        if not slurm_jobid:
            raise ValueError("SLURM_JOBID environment variable is not set.")

        master_port = str(10000 + int(slurm_jobid[-4:]))

    # Print the results for confirmation
    LOG.debug(f"MASTER_ADDR: {master_addr}")
    LOG.debug(f"MASTER_PORT: {master_port}")

    return master_addr, master_port

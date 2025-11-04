import os
import sys
import torch
import numpy as np
from pathlib import Path
import xarray as xr
from einops import rearrange
from icecream import ic
from anemoi.training.distributed.strategy import DDPGroupStrategy

import argparse
import torch.distributed as dist
import torch.multiprocessing as mp  # For launching processes
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from tqdm import tqdm
import datetime
import socket
from dataclasses import dataclass
from anemoi.training.diagnostics.local_inference.plot_predictions import (
    LocalInferencePlotter,
)
import time
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import subprocess
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.diagnostics.local_inference.sharding import (
    get_parallel_info,
    init_parallel,
    init_network,
)
from anemoi.training.diagnostics.local_inference.data_processing import (
    WeatherDataBatch,
    process_residuals,
    tensors_to_numpy,
    create_xarray_dataset,
)
from scipy.sparse import load_npz

import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_config_from_ckpt(
    dir_exp,
    name_exp,
    name_ckpt,
    config_dir="/home/ecm5702/dev/anemoi-config",
    config_name="hindcast_o320",
):
    logging.info(f"Preparing configs ...")
    checkpoint = torch.load(
        os.path.join(dir_exp, name_exp, name_ckpt),
        map_location=torch.device("cuda"),
        weights_only=False,
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, job_name="compose_config"):
        cfg = compose(config_name=config_name)
    cfg_ckpt = checkpoint["hyper_parameters"]["config"]
    cfg_ckpt.hardware.paths = cfg.hardware.paths
    return cfg_ckpt, checkpoint, cfg


def prepare_interface_model(
    config_checkpoint, checkpoint, data_indices, statistics, graph_data, device
):
    logging.info(f"Preparing model ...")
    truncation_data = {}

    truncation_data["down"] = load_npz(
        Path(
            config_checkpoint.hardware.paths.truncation,
            config_checkpoint.hardware.files.truncation,
        ),
    )
    truncation_data["up"] = load_npz(
        Path(
            config_checkpoint.hardware.paths.truncation,
            config_checkpoint.hardware.files.truncation_inv,
        ),
    )
    interface = AnemoiModelInterface(
        config=config_checkpoint,
        graph_data=graph_data,
        statistics=statistics,
        data_indices=data_indices,
        metadata=checkpoint["hyper_parameters"]["metadata"],
        truncation_data=truncation_data,
    )
    interface = interface.to(device)
    return interface


def prepare_datamodule(config_checkpoint, graph_data):
    logging.info(f"Preparing datamodule ...")

    from anemoi.training.data.datamodule import DownscalingAnemoiDatasetsDataModule

    datamodule = DownscalingAnemoiDatasetsDataModule(config_checkpoint, graph_data)
    return datamodule


def process_residuals(interface, x_in, y, samples, N_samples, N_members):
    x_in_interp_to_hres = interface.model._interpolate_to_high_res(x_in[:, 0, 0, ...])[
        :, None, None, ...
    ]
    y_residuals = y - x_in_interp_to_hres
    for i_sample in range(N_samples):
        for j_member in range(N_members):
            print(i_sample, j_member)
            x_interpol_to_hres_i = interface.model._interpolate_to_high_res(
                x_in[i_sample : i_sample + 1, 0, 0, ...]
            )[:, None, None, ...]
            samples[i_sample][j_member]["y_pred"] = (
                x_interpol_to_hres_i
                + samples[i_sample][j_member]["y_pred_residuals"].cpu()
            )
    return y_residuals, samples


@dataclass
class SampleSaver:
    dir_exp: "/home/ecm5702/scratch/aifs/checkpoint/"
    name_exp: "099e7dcdeca248198373d7397127edd5"
    name_ckpt: "last.ckpt"
    N_members: 3
    N_samples: 2
    return_intermediate: False

    def __post_init__(self) -> None:
        ### Prepare sharding
        self.global_rank, self.local_rank, self.world_size = get_parallel_info()
        self.device = f"cuda:{self.local_rank}"
        print(
            f"Running on global rank {self.global_rank} and local rank {self.local_rank} out of {self.world_size}"
        )
        torch.cuda.set_device(self.local_rank)

        self.model_comm_group = init_parallel(
            device=self.device, global_rank=self.global_rank, world_size=self.world_size
        )

        ### Checkpoint, graph, datamodule, model
        self.config_checkpoint, self.checkpoint, new_config = prepare_config_from_ckpt(
            self.dir_exp, self.name_exp, self.name_ckpt
        )
        graph_data = torch.load(
            "/home/ecm5702/scratch/aifs/graphs/o96_o320_icosahedral_r6_multiscale_h1_s6-1encoder.pt",
            weights_only=False,
        )
        self.datamodule = prepare_datamodule(new_config, graph_data)
        self.interface = prepare_interface_model(
            self.config_checkpoint,
            self.checkpoint,
            self.datamodule.data_indices,
            self.datamodule.statistics,
            graph_data,
            self.device,
        )

        ### Prepare data batch
        self.data_batch = WeatherDataBatch(self.datamodule.ds_valid)
        self.data_batch.prepare(idx=0, N_samples=self.N_samples)
        self.data_batch.prepare_miscellaneous()

    def sample(self, noise_scheduler_params=None, sampler_params=None):
        d_noise = {
            "schedule_type": "karras",
            "sigma_max": 100000.0,
            "sigma_min": 0.02,
            "rho": 7.0,
            "num_steps": 50,
        }
        d_samp = {
            "sampler": "heun",
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": 100000,
            "S_noise": 1.0,
        }
        if noise_scheduler_params:
            d_noise.update(noise_scheduler_params)
        if sampler_params:
            d_samp.update(sampler_params)
        self.samples = [None] * self.N_samples
        with torch.autocast(device_type="cuda"):
            for idx_sample in range(self.N_samples):
                logging.info(
                    f"Predicting sample {idx_sample}, Current GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                )
                self.samples[idx_sample] = [None] * self.N_members
                for idx_member in range(self.N_members):
                    logging.info(f"Sample {idx_sample}: Predicting member {idx_member}")
                    r = self.interface.predict_step(
                        self.data_batch.x_in[idx_sample].clone().to(self.device),
                        self.data_batch.x_in_hres[idx_sample].clone().to(self.device),
                        noise_scheduler_params=d_noise,
                        sampler_params=d_samp,
                        model_comm_group=self.model_comm_group,
                    )

                    self.samples[idx_sample][idx_member] = {"y_pred_residuals": r}

    def save_sampling(self, name_predictions_file):

        self.data_batch.x_in = self.data_batch.x_in.cpu()
        self.data_batch.y = self.data_batch.y.cpu()

        self.data_batch.y_residuals, self.samples = tensors_to_numpy(
            process_residuals(
                self.interface,
                self.data_batch.x_in,
                self.data_batch.y,
                self.samples,
                self.N_samples,
                self.N_members,
            )
        )

        self.data_batch.y_pred = np.array(
            [
                [member["y_pred"].squeeze() for member in sample]
                for sample in self.samples
            ]
        )
        self.data_batch.y_pred_residuals = np.array(
            [
                [member["y_pred_residuals"].squeeze() for member in sample]
                for sample in self.samples
            ]
        )

        """
        if self.return_intermediate:
            data_batch.intermediates = np.array(
                [
                    [
                        np.stack(
                            [step.squeeze() for step in member["intermediate_states"]],
                            axis=0,
                        )
                        for member in sample
                    ]
                    for sample in samples
                ]
            )
        """

        self.data_batch.prepare_miscellaneous()

        ds = create_xarray_dataset(
            self.data_batch,
            self.N_samples,
            self.N_members,
            self.config_checkpoint,
            self.datamodule.data_indices,
            # return_intermediate=return_intermediate,
        )

        # Add synchronization barrier before saving
        if self.model_comm_group is not None:
            torch.distributed.barrier(group=self.model_comm_group)

        # Only save from rank 0 process to avoid conflicts
        if self.global_rank == 0:
            predictions_path = os.path.join(
                self.dir_exp, self.name_exp, name_predictions_file
            )
            if os.path.exists(predictions_path):
                os.remove(predictions_path)
            ds.to_netcdf(predictions_path)
            logging.info(f"Predictions saved at {predictions_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference and save predictions.")
    parser.add_argument(
        "--name_exp", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--N_members", type=int, default=3, help="Number of ensemble members."
    )
    parser.add_argument("--nsteps", type=int, default=20, help="Number of steps.")
    parser.add_argument(
        "--N_samples", type=int, default=2, help="Number of samples to predict."
    )
    args = parser.parse_args()
    name_ckpt = "last.ckpt"

    if os.environ["HPC"] == "atos":
        dir_exp = "/home/ecm5702/scratch/aifs/checkpoint/"
    elif os.environ["HPC"] == "leo":
        dir_exp = "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/checkpoint"
    elif os.environ["HPC"] == "marenostrum":
        dir_exp = "/home/ecm/ecm800825/outputs/checkpoint"
    else:
        raise ValueError(f"Unknown HPC: {os.environ['HPC']}")

    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint directory: {dir_exp}")

    sample_saver = SampleSaver(
        dir_exp=dir_exp,
        name_exp=args.name_exp,
        name_ckpt=name_ckpt,
        N_members=args.N_members,
        N_samples=args.N_samples,
        return_intermediate=False,
    )
    sample_saver.sample(noise_scheduler_params={"num_steps": args.nsteps})
    name_predictions_file = "predictions.nc"
    sample_saver.save_sampling(name_predictions_file=name_predictions_file)
    logging.info("Waiting for all processes for 2mn before plotting")
    time.sleep(
        120
    )  # to make sure all processes are ready and predictions.nc is well saved
    lip = LocalInferencePlotter(dir_exp, args.name_exp, name_predictions_file)
    lip.save_plot(lip.regions, list_model_variables=["x", "y", "y_pred_0", "y_pred_1"])


# srun --partition=gpu --nodes=1 --gpus-per-node=2 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=2 python /home/ecm5702/dev/anemoi-training/src/anemoi/training/diagnostics/local_inference/save_sampling.py --name_exp 6aa131c4affa43e78660977b91bd3583

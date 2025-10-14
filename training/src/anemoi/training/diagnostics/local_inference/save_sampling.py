import os
import sys
import torch
import numpy as np
import xarray as xr
from einops import rearrange
from icecream import ic
from anemoi.training.distributed.strategy import DDPGroupStrategy
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.train.downscaler import GraphDownscaler
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp  # For launching processes
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from tqdm import tqdm
import datetime
import socket
from dataclasses import dataclass
from anemoi.training.data.ds_dataset import DownscalingDataset
from anemoi.training.diagnostics.local_inference.plot_predictions import (
    LocalInferencePlotter,
)
import time
import subprocess

from anemoi.training.diagnostics.local_inference.sharding import (
    __get_parallel_info,
    __init_parallel,
    __init_network,
)
from anemoi.training.diagnostics.local_inference.data_processing import (
    WeatherDataBatch,
    process_residuals,
    tensors_to_numpy,
    create_xarray_dataset,
)


import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_checkpoint(path_ckpt):
    cp_full = torch.load(
        path_ckpt, map_location=torch.device("cuda"), weights_only=False
    )
    config = cp_full["hyper_parameters"]["config"]
    config.diagnostics.log.mlflow.enabled = False
    return cp_full, config


# Initialize trainer and downscaler
def initialize_trainer_and_downscaler(config, cp_full):
    trainer = AnemoiTrainer(config)
    kwargs = {
        "config": trainer.config,
        "data_indices": trainer.data_indices,
        "graph_data": trainer.graph_data,
        "metadata": trainer.metadata,
        "statistics": trainer.datamodule.statistics,
    }
    downscaler = GraphDownscaler(**kwargs)
    downscaler.load_state_dict(cp_full["state_dict"])
    return trainer, downscaler


# Main function
def main(
    N_members,
    name_exp,
    name_predictions_file="predictions.nc",
    use_ds_predict=True,
    nsteps=20,
    N_samples=2,
):

    global_rank, local_rank, world_size = __get_parallel_info()
    device = f"cuda:{local_rank}"
    print(
        f"Running on global rank {global_rank} and local rank {local_rank} out of {world_size}"
    )
    torch.cuda.set_device(local_rank)

    model_comm_group = __init_parallel(
        device=device, global_rank=global_rank, world_size=world_size
    )

    cp_full, config = load_checkpoint(PATH_CKPT)
    trainer, downscaler = initialize_trainer_and_downscaler(config, cp_full)

    N_members = 1
    idx = 0
    nsteps = 1
    return_intermediate = False

    dataset_choice = (
        trainer.datamodule.ds_predict if use_ds_predict else trainer.datamodule.ds_test
    )
    data_batch = WeatherDataBatch(dataset_choice)
    data_batch.prepare(idx=idx, N_samples=N_samples)
    data_batch.prepare_miscellaneous()

    downscaler.model = downscaler.model.to(device)

    samples = [None] * N_samples
    with torch.autocast(device_type="cuda"):
        for idx_sample in range(0, N_samples):
            print(
                f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            logging.info(f"Predicting sample {idx_sample}")
            samples[idx_sample] = [None] * N_members
            for idx_member in range(0, N_members):
                logging.info(f"Sample {idx_sample}: Predicting member {idx_member}")
                samples[idx_sample][idx_member] = (
                    downscaler.model.predict_step_with_intermediates(
                        data_batch.x_in[idx_sample].clone().to(device),
                        data_batch.x_in_hres[idx_sample].clone().to(device),
                        nsteps=nsteps,
                        return_intermediate=return_intermediate,
                        model_comm_group=model_comm_group,
                    )
                )

    ic("Prediction made")

    # add all variables for xarray into data_batch
    data_batch.x_in = data_batch.x_in.cpu()
    data_batch.y = data_batch.y.cpu()

    if config.training.predict_residuals:
        y_residuals, samples = tensors_to_numpy(
            process_residuals(
                downscaler, data_batch.x_in, data_batch.y, samples, N_samples, N_members
            )
        )
    else:
        y_residuals = None
    data_batch.y_residuals = y_residuals

    data_batch.y_pred = np.array(
        [[member["y_pred"].squeeze() for member in sample] for sample in samples]
    )
    if config.training.predict_residuals:
        data_batch.y_pred_residuals = np.array(
            [
                [member["residuals_prediction"].squeeze() for member in sample]
                for sample in samples
            ]
        )

    if return_intermediate:
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

    data_batch.prepare_miscellaneous()

    ds = create_xarray_dataset(
        data_batch,
        N_samples,
        N_members,
        config,
        downscaler,
        return_intermediate=return_intermediate,
    )

    # Add synchronization barrier before saving
    if model_comm_group is not None:
        torch.distributed.barrier(group=model_comm_group)

    # Only save from rank 0 process to avoid conflicts
    if global_rank == 0:
        predictions_path = os.path.join(DIR_EXP, args.name_exp, name_predictions_file)
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
    parser.add_argument(
        "--use_ds_predict",
        type=lambda x: x.lower() in ['true', '1', 'yes', 'y'],
        default=True,
        help="Use ds_predict dataset. Defaults to True."
    )   
    args = parser.parse_args()

    name_predictions_file = (
        "predictions.nc" if args.use_ds_predict else "predictions_test.nc"
    )

    # Constants
    if os.environ["HPC"] == "atos":
        DIR_EXP = "/home/ecm5702/scratch/aifs/checkpoint/"
    elif os.environ["HPC"] == "leo":
        DIR_EXP = "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/checkpoint"
    elif os.environ["HPC"] == "marenostrum":
        DIR_EXP = "/home/ecm/ecm800825/outputs/checkpoint"
    else:
        raise ValueError(f"Unknown HPC: {os.environ['HPC']}")

    NAME_CKPT = "last.ckpt"
    PATH_CKPT = os.path.join(DIR_EXP, args.name_exp, NAME_CKPT)

    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint directory: {DIR_EXP}")

    main(
        N_members=args.N_members,
        name_exp=args.name_exp,
        nsteps=args.nsteps,
        N_samples=args.N_samples,
        use_ds_predict=args.use_ds_predict,
        name_predictions_file=name_predictions_file,
    )
    logging.info("Waiting for all processes for 2mn before plotting")
    time.sleep(
        120
    )  # to make sure all processes are ready and predictions.nc is well saved
    lip = LocalInferencePlotter(DIR_EXP, args.name_exp, name_predictions_file)
    lip.save_plot(lip.regions)


# srun --partition=gpu --nodes=1 --gpus-per-node=2 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=2 python /home/ecm5702/dev/anemoi-training/src/anemoi/training/diagnostics/local_inference/save_sampling.py --name_exp 6aa131c4affa43e78660977b91bd3583

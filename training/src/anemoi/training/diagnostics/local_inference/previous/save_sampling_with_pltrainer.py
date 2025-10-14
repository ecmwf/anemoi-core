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
import logging

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append("/home/ecm5702/dev/inference")


# Load checkpoint and configuration
def load_checkpoint(path_ckpt, frequency="180"):
    cp_full = torch.load(path_ckpt, map_location=torch.device("cuda"))
    config = cp_full["hyper_parameters"]["config"]
    config.dataloader.batch_size.predict = 1
    config.dataloader.predict.frequency = frequency
    config.diagnostics.log.mlflow.enabled = False
    config.dataloader.num_workers.predict = 4
    config.dataloader.num_workers.test = 4
    config.dataloader.prefetch_factor = 2
    # config.hardware.files.residual_statistics = "o320_dict.npy"
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
    downscaler = downscaler.cuda()
    return trainer, downscaler


# Process residuals
def process_residuals(downscaler, x_in, y, samples, N_samples, N_members):
    x_in_interp_to_hres = downscaler.model.interpolate_down(
        x_in[:, 0, 0, ...], grad_checkpoint=False
    )[:, None, None, ...]
    downscaler.x_in_matching_channel_indices = (
        downscaler.x_in_matching_channel_indices.to(x_in_interp_to_hres.device)
    )
    y_residuals = y - x_in_interp_to_hres[..., downscaler.x_in_matching_channel_indices]
    for i_sample in range(N_samples):
        for j_member in range(N_members):
            x_interpol_to_hres_i = downscaler.model.interpolate_down(
                x_in[i_sample : i_sample + 1, 0, 0, ...], grad_checkpoint=False
            )[:, None, None, ...][..., downscaler.x_in_matching_channel_indices]
            samples[i_sample]["prediction"][j_member]["y_pred"] = (
                x_interpol_to_hres_i
                + samples[i_sample]["prediction"][j_member]["residuals_prediction"]
            )
    return y_residuals, samples


# Convert tensors to numpy
def tensors_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, list):
        if all(isinstance(item, np.ndarray) for item in obj):
            return obj
        return [tensors_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(tensors_to_numpy(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: tensors_to_numpy(value) for key, value in obj.items()}
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
    return obj


# Extract filtered input from output
def extract_filtered_input_from_output(
    input_weather_states, input_name_to_index, output_name_to_index
):
    common_weather_states = set(input_name_to_index.keys()) & set(
        output_name_to_index.keys()
    )
    filtered_keys_sorted = sorted(
        common_weather_states, key=lambda k: output_name_to_index[k]
    )
    filtered_indices_sorted = [input_name_to_index[key] for key in filtered_keys_sorted]
    filtered_input_weather_states = input_weather_states[..., filtered_indices_sorted]
    filtered_input_name_to_index = {
        key: i for i, key in enumerate(filtered_keys_sorted)
    }
    return filtered_input_weather_states, filtered_input_name_to_index


# Create xarray dataset
def create_xarray_dataset(
    samples,
    x_in,
    y,
    y_residuals,
    trainer,
    N_samples,
    N_members,
    config,
    downscaler,
    return_intermediate=False,
):
    predict_loader = trainer.datamodule.predict_dataloader(num_workers=4)
    dates_samples = predict_loader.dataset.data.dates[
        predict_loader.dataset.valid_date_indices
    ]
    longitudes_lres = predict_loader.dataset.data.longitudes[0]
    latitudes_lres = predict_loader.dataset.data.latitudes[0]
    longitudes_hres = predict_loader.dataset.data.longitudes[2]
    latitudes_hres = predict_loader.dataset.data.latitudes[2]

    y_pred = np.array(
        [
            [member["y_pred"].squeeze() for member in sample["prediction"]]
            for sample in samples
        ]
    )
    if config.training.predict_residuals:
        y_pred_residuals = np.array(
            [
                [
                    member["residuals_prediction"].squeeze()
                    for member in sample["prediction"]
                ]
                for sample in samples
            ]
        )
    x_for_ds = np.array(x_in)
    y_for_ds = np.array(y)
    if config.training.predict_residuals:
        y_residuals_for_ds = np.array(y_residuals)

    if return_intermediate:
        intermediates = np.array(
            [
                [
                    np.stack(
                        [step.squeeze() for step in member["intermediate_states"]],
                        axis=0,
                    )
                    for member in sample["prediction"]
                ]
                for sample in samples
            ]
        )

    x_for_ds, filtered_input_name_to_index = extract_filtered_input_from_output(
        x_for_ds,
        downscaler.data_indices.data.input[0].name_to_index,
        downscaler.data_indices.model.output.name_to_index,
    )
    if len(x_for_ds.shape) != 3:
        x_for_ds = x_for_ds[:, 0, 0, ...]
        y_for_ds = y_for_ds[:, 0, 0, ...]
        if config.training.predict_residuals:
            y_residuals_for_ds = y_residuals_for_ds[:, 0, 0, ...]

    ds = xr.Dataset(
        {
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                y_pred,
            ),
            "x": (["sample", "grid_point_lres", "weather_state"], x_for_ds),
            "y": (["sample", "grid_point_hres", "weather_state"], y_for_ds),
            "date": (["sample"], dates_samples),
            "lon_lres": (["grid_point_lres"], longitudes_lres),
            "lat_lres": (["grid_point_lres"], latitudes_lres),
            "lon_hres": (["grid_point_hres"], longitudes_hres),
            "lat_hres": (["grid_point_hres"], latitudes_hres),
        },
        coords={
            "sample": range(N_samples),
            "ensemble_member": range(N_members),
            "grid_point_lres": range(longitudes_lres.shape[0]),
            "grid_point_hres": range(longitudes_hres.shape[0]),
            "weather_state": list(
                downscaler.data_indices.model.output.name_to_index.keys()
            ),
        },
    )

    ds.x.attrs["lon"] = "lon_lres"
    ds.x.attrs["lat"] = "lat_lres"
    ds.y.attrs["lon"] = "lon_hres"
    ds.y.attrs["lat"] = "lat_hres"
    ds.y_pred.attrs["lon"] = "lon_hres"
    ds.y_pred.attrs["lat"] = "lat_hres"
    ds["lon_hres"] = ((ds.lon_hres + 180) % 360) - 180
    ds["lon_lres"] = ((ds.lon_lres + 180) % 360) - 180

    if return_intermediate:
        ds["inter_state"] = (
            [
                "sample",
                "ensemble_member",
                "sampling_step",
                "grid_point_hres",
                "weather_state",
            ],
            intermediates,
        )

    for i in range(N_members):
        ds[f"y_pred_{i}"] = ds["y_pred"].sel(ensemble_member=i)
        ds[f"y_pred_{i}_diff"] = ds[f"y_pred_{i}"] - ds["y"]
        ds[f"y_pred_{i}"].attrs["lon"] = "lon_hres"
        ds[f"y_pred_{i}"].attrs["lat"] = "lat_hres"
        ds[f"y_pred_{i}_diff"].attrs["lon"] = "lon_hres"
        ds[f"y_pred_{i}_diff"].attrs["lat"] = "lat_hres"

    if config.training.predict_residuals:
        ds["y_pred_residuals"] = (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            y_pred_residuals,
        )
        ds["y_residuals"] = (
            ["sample", "grid_point_hres", "weather_state"],
            y_residuals_for_ds,
        )

    if config.training.predict_residuals:
        ds["y_pred_residuals"] = (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            y_pred_residuals,
        )
        ds["y_residuals"] = (
            ["sample", "grid_point_hres", "weather_state"],
            y_residuals_for_ds,
        )

    return ds


# Main function
def main(
    name_exp,
    N_members,
    num_gpus_per_model,
    num_nodes,
    num_gpus_per_node,
    frequency="500",
):

    cp_full, config = load_checkpoint(PATH_CKPT, frequency=frequency)
    config.hardware.num_nodes = num_nodes
    config.hardware.num_gpus_per_node = num_gpus_per_node
    config.hardware.num_gpus_per_model = num_gpus_per_model
    trainer, downscaler = initialize_trainer_and_downscaler(config, cp_full)

    downscaler.multi_samples_pred = True
    downscaler.inference_nsteps = 20
    downscaler.return_intermediate = False
    downscaler.N_members = N_members

    # generate samples here
    inference_loader = trainer.datamodule.predict_dataloader(num_workers=4)
    pl_trainer = pl.Trainer(
        accelerator=trainer.accelerator,  # Needed for hardware acceleration (GPU/CPU)
        strategy=DDPGroupStrategy(
            num_gpus_per_model
        ),  # Required for model parallelization strategy
        devices=num_gpus_per_node,  # Specify GPU distribution
        num_nodes=num_nodes,  # Required for multi-node setup
        precision=trainer.config.training.precision,  # Keep same precision as training
        deterministic=trainer.config.training.deterministic,  # Maintain deterministic behavior
        enable_progress_bar=trainer.config.diagnostics.enable_progress_bar,  # Optional but useful
    )
    # samples is a list of N_samples
    # each element is a list of N_members
    # each element is a dict with keys "y_pred", "residuals_prediction", "intermediate_states"

    with torch.autocast(device_type="cuda") and torch.no_grad():
        samples = pl_trainer.predict(
            downscaler,
            inference_loader,
        )

    N_samples = len(samples)
    # x_in = torch.cat([sample["input"] for sample in samples], dim=0)
    # y = torch.cat([sample["truth"] for sample in samples], dim=0)
    x_in = np.concatenate([sample["input"] for sample in samples], axis=0)
    y = np.concatenate([sample["truth"] for sample in samples], axis=0)
    if config.training.predict_residuals:
        y_residuals, samples = tensors_to_numpy(
            process_residuals(downscaler, x_in, y, samples, N_samples, N_members)
        )
    else:
        y_residuals = None

    ds = create_xarray_dataset(
        tensors_to_numpy(samples),
        tensors_to_numpy(x_in),
        tensors_to_numpy(y),
        y_residuals,
        trainer,
        N_samples,
        N_members,
        config,
        downscaler,
        return_intermediate=downscaler.return_intermediate,
    )

    ds.to_netcdf(os.path.join(DIR_EXP, name_exp, "predictions.nc"))
    print(os.path.join(DIR_EXP, name_exp, "predictions.nc"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference and save predictions.")
    parser.add_argument(
        "--name_exp", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--N_members", type=int, default=3, help="Number of ensemble members."
    )
    args = parser.parse_args()

    # Constants
    if os.environ["HPC"] == "atos":
        DIR_EXP = "/home/ecm5702/scratch/aifs/checkpoint/"
    elif os.environ["HPC"] == "leo":
        DIR_EXP = "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/checkpoint"
    else:
        raise ValueError(f"Unknown HPC: {os.environ['HPC']}")

    NAME_CKPT = "last.ckpt"
    PATH_CKPT = os.path.join(DIR_EXP, args.name_exp, NAME_CKPT)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint directory: {DIR_EXP}")

    os.environ["NCCL_TIMEOUT"] = "3600"

    main(
        name_exp=args.name_exp,
        N_members=args.N_members,
        num_gpus_per_model=8,
        num_nodes=2,
        num_gpus_per_node=4,
        frequency="180",
    )

    # Example run
    # srun --partition=boost_usr_prod --account=DestE_340_25 --gpus-per-node=1 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=1  python /leonardo/home/userexternal/jdumontl/dev/anemoi-training/src/anemoi/training/diagnostics/nb_plots/plot_local_pred.py --name_exp 7c321d69dc5f4f2f81935ae3f7606f9b --N_samples 2 --N_members 3
    # srun --partition=boost_usr_prod --account=DestE_340_25 --gpus-per-node=1 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=1  python /leonardo/home/userexternal/jdumontl/dev/anemoi-training/src/anemoi/training/diagnostics/nb_plots/plot_local_pred.py --name_exp 50842e050f034a739ef97b3049a2cd82 --N_samples 2 --N_members 3
    # srun --partition=gpu --nodes=1 --gpus-per-node=2 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=2  python /home/ecm5702/dev/anemoi-training/src/anemoi/training/diagnostics/nb_plots/save_sampling_2.py --name_exp 6aa131c4affa43e78660977b91bd3583 --N_members 1
    # srun --partition=gpu --nodes=2 --gpus-per-node=4 -c 8 --mem=128G -t 00:30:0 --ntasks-per-node=4  python /home/ecm5702/dev/anemoi-training/src/anemoi/training/diagnostics/nb_plots/save_sampling.py --name_exp 30040c70a9ca4c8092ca1688a2f9c190 --N_samples 2 --N_members 3

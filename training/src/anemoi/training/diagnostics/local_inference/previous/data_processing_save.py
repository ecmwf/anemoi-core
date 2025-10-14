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
from anemoi.training.data.ds_dataset import DownscalingDataset
from dataclasses import dataclass

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append("/home/ecm5702/dev/inference")


@dataclass
class WeatherDataBatch:
    dataset: DownscalingDataset
    device: str = "cuda"

    def prepare(self, idx: int, N_samples: int):
        x_in = self.dataset.data[idx : idx + N_samples][0]
        x_in_hres = self.dataset.data[idx : idx + N_samples][1]
        y = self.dataset.data[idx : idx + N_samples][2]

        self.x_in = self._process_tensor(x_in)[:, None, ...]
        self.x_in_hres = self._process_tensor(x_in_hres)[:, None, ...]
        self.y = self._process_tensor(y)[:, None, ...]

        self.date = self.dataset.data.dates[idx : idx + N_samples]

        ic(self.x_in.shape, self.x_in_hres.shape, self.y.shape)

    def prepare_miscellaneous(self):
        self.lon_lres = self.dataset.data.longitudes[0]
        self.lat_lres = self.dataset.data.latitudes[0]
        self.lon_hres = self.dataset.data.longitudes[2]
        self.lat_hres = self.dataset.data.latitudes[2]

    def _process_tensor(self, tensor):
        tensor = rearrange(
            tensor,
            "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
        )
        return torch.from_numpy(tensor)

    def convert_to_numpy(self):
        self.x_in = tensors_to_numpy(self.x_in)
        self.x_in_hres = tensors_to_numpy(self.x_in_hres)
        self.y = tensors_to_numpy(self.y)


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
            samples[i_sample][j_member]["y_pred"] = (
                x_interpol_to_hres_i
                + samples[i_sample][j_member]["residuals_prediction"]
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
    data_batch,
    N_samples,
    N_members,
    config,
    downscaler,
    return_intermediate=False,
):

    if data_batch.x_in.shape[-1] != data_batch.y.shape[-1]:
        data_batch.x_in, filtered_input_name_to_index = (
            extract_filtered_input_from_output(
                data_batch.x_in,
                downscaler.data_indices.data.input[0].name_to_index,
                downscaler.data_indices.model.output.name_to_index,
            )
        )
    if len(data_batch.x_in.shape) != 3:
        data_batch.x_in = data_batch.x_in[:, 0, 0, ...]
        data_batch.y = data_batch.y[:, 0, 0, ...]
        if config.training.predict_residuals:
            data_batch.y_residuals = data_batch.y_residuals[:, 0, 0, ...]

    ds = xr.Dataset(
        {
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                data_batch.y_pred,
            ),
            "x": (["sample", "grid_point_lres", "weather_state"], data_batch.x_in),
            "y": (["sample", "grid_point_hres", "weather_state"], data_batch.y),
            "date": (["sample"], data_batch.date),
            "lon_lres": (["grid_point_lres"], data_batch.lon_lres),
            "lat_lres": (["grid_point_lres"], data_batch.lat_lres),
            "lon_hres": (["grid_point_hres"], data_batch.lon_hres),
            "lat_hres": (["grid_point_hres"], data_batch.lat_hres),
        },
        coords={
            "sample": range(N_samples),
            "ensemble_member": range(N_members),
            "grid_point_lres": range(data_batch.lon_lres.shape[0]),
            "grid_point_hres": range(data_batch.lon_hres.shape[0]),
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
            data_batch.intermediates,
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
            data_batch.y_pred_residuals,
        )
        ds["y_residuals"] = (
            ["sample", "grid_point_hres", "weather_state"],
            data_batch.y_residuals,
        )

    return ds

from logging import config
import torch
import os
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from scipy.sparse import load_npz
from pathlib import Path
from dataclasses import asdict, is_dataclass


class ObjectFromCheckpointLoader:
    def __init__(self, dir_exp, name_exp, name_ckpt):
        self.dir_exp = dir_exp
        self.name_exp = name_exp
        self.name_ckpt = name_ckpt
        self.checkpoint, self.config_checkpoint = get_checkpoint(
            dir_exp, name_exp, name_ckpt
        )
        # Keep checkpoint-native data/dataloader settings, only adapt local paths.
        local_paths_config = instantiate_config()
        self.config_checkpoint = adapt_config_hpc(
            self.config_checkpoint, local_paths_config
        )
        self.config_for_datamodule = to_omegaconf(self.config_checkpoint)

    def load(self):
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_model = torch.load(
            os.path.join(self.dir_exp, self.name_exp, "inference-" + self.name_ckpt),
            map_location=map_location,
            weights_only=False,
        )
        self.graph_data = self.inference_model.graph_data
        self.truncation_data = self.inference_model.truncation_data

        self.datamodule = get_datamodule(self.config_for_datamodule, self.graph_data)
        self.interface = get_interface(
            self.config_checkpoint,
            self.datamodule,
            self.graph_data,
            self.truncation_data,
            self.checkpoint,
        )
        self.downscaler = get_downscaler(
            self.dir_exp,
            self.name_exp,
            self.name_ckpt,
            self.checkpoint,
            self.config_checkpoint,
            self.graph_data,
            self.datamodule,
            self.truncation_data,
        )


def get_checkpoint(dir_exp, name_exp, name_ckpt):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        os.path.join(dir_exp, name_exp, name_ckpt),
        map_location=map_location,
        weights_only=False,
    )
    config_checkpoint = checkpoint["hyper_parameters"]["config"]
    return checkpoint, config_checkpoint


def adapt_config_hpc(config_checkpoint, config):
    # config_checkpoint.hardware.paths = config.hardware.paths
    config_checkpoint.hardware.paths = OmegaConf.to_container(
        config.hardware.paths, resolve=True
    )
    return config_checkpoint


def instantiate_config(
    anemoi_config_dir="/home/ecm5702/dev/anemoi-config", config_name="hindcast_o320"
):
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=anemoi_config_dir, job_name="compose_config")
    config = compose(config_name=config_name)
    return config


def _coerce_to_container(obj):
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=False)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (dict, list, tuple)):
        return obj
    if hasattr(obj, "__dict__"):
        return {
            k: v
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return obj


def to_omegaconf(obj):
    if OmegaConf.is_config(obj):
        return obj
    return OmegaConf.create(_coerce_to_container(obj))


def get_datamodule(config, graph_data):
    # Compatibility: old stacks expose AnemoiDatasetsDataModule, newer stacks
    # expose DownscalingAnemoiDatasetsDataModule.
    try:
        from anemoi.training.data.datamodule import DownscalingAnemoiDatasetsDataModule as DataModuleCls
    except ImportError:
        from anemoi.training.data.datamodule import AnemoiDatasetsDataModule as DataModuleCls

    try:
        datamodule = DataModuleCls(config, graph_data)
    except TypeError:
        datamodule = DataModuleCls(config)
    # data_indices = datamodule.data_indices
    # statistics = datamodule.statistics
    # supporting_arrays = datamodule.supporting_arrays
    return datamodule


def get_interface(
    config_checkpoint, datamodule, graph_data, truncation_data, checkpoint
):
    from anemoi.models.interface import AnemoiModelInterface

    interface = AnemoiModelInterface(
        config=config_checkpoint,
        graph_data=graph_data,
        statistics=datamodule.statistics,
        data_indices=datamodule.data_indices,
        metadata=checkpoint["hyper_parameters"]["metadata"],
        truncation_data=truncation_data,
    )
    return interface


def get_downscaler(
    dir_exp,
    name_exp,
    name_ckpt,
    checkpoint,
    config_checkpoint,
    graph_data,
    datamodule,
    truncation_data,
):
    from anemoi.training.train.tasks.downscaler import GraphDiffusionDownscaler

    kwargs = {
        "config": config_checkpoint,
        "data_indices": datamodule.data_indices,
        "graph_data": graph_data,
        "metadata": checkpoint["hyper_parameters"]["metadata"],
        "statistics": datamodule.statistics,
        "statistics_tendencies": datamodule.statistics,
        "supporting_arrays": datamodule.supporting_arrays,
        "truncation_data": truncation_data,
    }

    downscaler = GraphDiffusionDownscaler.load_from_checkpoint(
        os.path.join(dir_exp, name_exp, name_ckpt), strict=False, **kwargs
    )
    # downscaler = downscaler.to(device)
    return downscaler


"""
device = "cuda"
object_loader = ObjectFromCheckpointLoader(dir_exp, name_exp, name_ckpt)
object_loader.config_for_datamodule.dataloader.validation.frequency = "50h"
object_loader.load()
# object_loader.config_checkpoint: modify here if desired

datamodule = object_loader.datamodule
interface = object_loader.interface.to(device)
downscaler = object_loader.downscaler.to(device)
graph_data = object_loader.graph_data
inference_model = object_loader.inference_model.to(device)

"""
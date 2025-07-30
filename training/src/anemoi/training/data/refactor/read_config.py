import os
from typing import Dict

from omegaconf import DictConfig
from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_OBS = os.path.join(BASE_DIR, "configs", "config_obs.yaml")
CONFIG_DOWNSCALING = os.path.join(BASE_DIR, "configs", "config_downscaling.yaml")
CONFIG_MULTIPLE = os.path.join(BASE_DIR, "configs", "config_multiple.yaml")


def get_example(which: str) -> Dict:
    if which == "obs":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_OBS), resolve=True)
    if which == "downscaling":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_DOWNSCALING), resolve=True)
    if which == "multiple":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_MULTIPLE), resolve=True)
    raise ValueError("Only supportings examples: obs, downscaling, multiple.")


def get_data_config_dict(data, which: str) -> Dict:
    return get_example(which)["data"]


def get_sample_config_dict(sample: DictConfig, which: str) -> Dict:
    return get_example(which)["sample"]


def convert_source(config, name: str) -> Dict:
    return {
        "tensor": [dict(variables=[f"{name}.{v}" for v in config["variables"]]), dict(offset=config["offset"])],
    }


def convert_sample_config(config) -> Dict:
    c = {}
    for n, cfgs in config.items():
        c[n] = {"dictionary": {source: convert_source(cfg, name=source) for source, cfg in cfgs.items()}}
    return {"dictionary": c}


def convert_dataset(dataset, group_name: str):
    if isinstance(dataset["dataset"], str):
        dataset["dataset"] = {"dataset": dataset["dataset"]}
    dataset["dataset"]["set_group"] = group_name
    return dataset


def convert_data_config(config) -> Dict:
    return {k: convert_dataset(v, group_name=k) for k, v in config.items()}


def get_config_dict(config: DictConfig) -> Dict:
    return dict(
        data=convert_data_config(config["data"]["data_handlers"]),
        sample=convert_sample_config(config["model"]["sample"]),
    )

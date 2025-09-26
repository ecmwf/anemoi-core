import os
from typing import Dict

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.utils.config import DotDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_OBS = os.path.join(BASE_DIR, "configs", "config_obs.yaml")
CONFIG_DOWNSCALING = os.path.join(BASE_DIR, "configs", "config_downscaling.yaml")
CONFIG_MULTIPLE = os.path.join(BASE_DIR, "configs", "config_multiple.yaml")

DATA_KEY = "data"
SAMPLE_KEY = "sample"


class TrainingAnemoiConfig(DotDict):
    EXAMPLE_FILE = None

    @classmethod
    def get_example(cls):
        return cls.from_yaml_file(cls.EXAMPLE_FILE)

    @classmethod
    def from_simple_config(cls, simple):
        raise ValueError(f"Not implemented yet for {cls.__name__}")


class ObsAnemoiConfig(TrainingAnemoiConfig):
    EXAMPLE_FILE = os.path.join(BASE_DIR, "configs", "config_obs.yaml")


class DownscalingAnemoiConfig(TrainingAnemoiConfig):
    EXAMPLE_FILE = os.path.join(BASE_DIR, "configs", "config_downscaling.yaml")

    @classmethod
    def from_simple_config(cls, simple):
        cls._simple_config = simple

        data = convert_data_config(simple["data"]["sources"])
        sample = convert_sample_config(simple["model"]["sample"])

        full = {DATA_KEY: data, SAMPLE_KEY: sample}
        return cls(full)


class GeneralAnemoiConfig(TrainingAnemoiConfig):
    @classmethod
    def get_example(cls):
        raise ValueError("No example for GeneralAnemoiConfig")


def get_example(which: str) -> Dict:
    if which == "obs":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_OBS), resolve=True)
    if which == "downscaling":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_DOWNSCALING), resolve=True)
    if which == "multiple":
        return OmegaConf.to_container(OmegaConf.load(CONFIG_MULTIPLE), resolve=True)
    raise ValueError("Only supportings examples: obs, downscaling, multiple.")


# use config.data instead
def get_data_config_dict(data, which: str) -> Dict:
    return get_example(which)["data"]


# use config.sample
def get_sample_config_dict(sample: DictConfig, which: str) -> Dict:
    return get_example(which)["sample"]


def convert_source(config, name: str) -> Dict:
    variables = config["variables"]
    container = dict(data_group=name, variables=variables, dimensions=["variables", "values"])
    offset = config.get("offset")

    if offset == "0h" or offset is None:  # i.e. no offset
        return dict(container=container)

    if isinstance(offset, str):
        return dict(offset=offset, container=container)

    if isinstance(offset, list) or isinstance(offset, ListConfig):  # TODO remove this 'if' and use classes
        return dict(
            merge_offset_as_first_dimension=[
                # for_each=[
                dict(offset=offset),
                dict(container=container),
            ],
        )

    raise ValueError(f"Invalid offset type {type(offset)}")


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
        data=convert_data_config(config["data"]["sources"]),
        sample=convert_sample_config(config["model"]["sample"]),
    )

"""Configuration utilities for handling dataset-specific configurations."""

from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict


def get_multiple_datasets_config(config: DictConfig) -> dict:
    """Get multiple datasets configuration for old configs.

    Use /'data/' as the default dataset name.
    """
    if "datasets" in config:
        if isinstance(config, dict):
            return config["datasets"]
        return config.datasets

    return OmegaConf.create({"data": config})

def get_dataset_data_config(config: DictConfig, dataset_name: str | None = None) -> DictConfig:
    """Get dataset-specific data configuration.

    Parameters
    ----------
    config : DictConfig
        The full configuration object
    dataset_name : str, optional
        Name of the dataset. If None, returns config.data as-is (single-dataset mode).

    Returns
    -------
    DictConfig
        Dataset-specific data configuration (the content that would be under config.data)

    Examples
    --------
    # Single-dataset mode (existing behavior)
    data_config = get_dataset_data_config(config)  # dataset_name=None
    # data_config contains forcing, diagnostic, processors, etc.

    # Multi-dataset mode
    era5_data_config = get_dataset_data_config(config, 'era5')
    # era5_data_config contains ERA5-specific forcing, diagnostic, etc.
    """
    # Single-dataset mode: return config.data as-is
    if dataset_name is None:
        return config.data

    # Multi-dataset mode: check for dataset-specific data config
    if hasattr(config, "data") and hasattr(config.data, "dataset_specific") and hasattr(config.data.dataset_specific, "datasets") and dataset_name in config.data.dataset_specific.datasets:

        dataset_config = config.data.dataset_specific.datasets[dataset_name]

        # Merge with default if it exists
        if hasattr(config.data, "default"):
            return OmegaConf.merge(config.data.default, dataset_config)
        return dataset_config

    # Fallback: use base config.data (backwards compatibility)
    return config.data


def get_dataset_scalers_config(config: DictConfig, dataset_name: str | None = None) -> DictConfig:
    """Get dataset-specific scalers configuration.

    Parameters
    ----------
    config : DictConfig
        The full configuration object
    dataset_name : str, optional
        Name of the dataset. If None, returns config.training.scalers as-is (single-dataset mode).

    Returns
    -------
    DictConfig
        Dataset-specific scalers configuration

    Examples
    --------
    # Single-dataset mode (existing behavior)
    scalers_config = get_dataset_scalers_config(config)  # dataset_name=None
    # scalers_config contains general_variable, nan_mask_weights, etc.

    # Multi-dataset mode
    era5_scalers_config = get_dataset_scalers_config(config, 'era5')
    # era5_scalers_config contains ERA5-specific scalers with inheritance from default
    """
    # Single-dataset mode: return config.training.scalers as-is
    if dataset_name is None:
        return config.training.scalers

    # Multi-dataset mode: check for dataset-specific scalers config
    if (
        hasattr(config, "training")
        and hasattr(config.training, "scalers")
        and hasattr(config.training.scalers, "datasets")
        and dataset_name in config.training.scalers.datasets
    ):

        dataset_config = config.training.scalers.datasets[dataset_name]

        # Merge with default if it exists
        if hasattr(config.training.scalers, "default"):
            return OmegaConf.merge(config.training.scalers.default, dataset_config)
        return dataset_config

    # Fallback: use base config.training.scalers (backwards compatibility)
    return config.training.scalers


def get_dataset_loss_and_metrics_config(config: DictConfig, dataset_name: str | None = None) -> DictConfig:
    """Get dataset-specific loss and metrics configuration.

    Parameters
    ----------
    config : DictConfig
        The full configuration object
    dataset_name : str, optional
        Name of the dataset. If None, returns legacy structure (single-dataset mode).

    Returns
    -------
    DictConfig
        Dataset-specific loss and metrics configuration containing 'training_loss' and 'validation_metrics'

    Examples
    --------
    # Single-dataset mode (existing behavior)
    loss_metrics_config = get_dataset_loss_and_metrics_config(config)  # dataset_name=None
    # loss_metrics_config.training_loss and loss_metrics_config.validation_metrics

    # Multi-dataset mode
    era5_config = get_dataset_loss_and_metrics_config(config, 'era5')
    # era5_config contains ERA5-specific training_loss and validation_metrics with inheritance
    """
    # Single-dataset mode: return legacy structure
    if dataset_name is None:
        return OmegaConf.create(
            {"training_loss": config.training.training_loss, "validation_metrics": config.training.validation_metrics},
        )

    # Multi-dataset mode: check for loss_and_metrics structure
    if (
        hasattr(config, "training")
        and hasattr(config.training, "loss_and_metrics")
        and hasattr(config.training.loss_and_metrics, "datasets")
        and dataset_name in config.training.loss_and_metrics.datasets
    ):

        dataset_config = config.training.loss_and_metrics.datasets[dataset_name]

        # Merge with default if it exists
        if hasattr(config.training.loss_and_metrics, "default"):
            return OmegaConf.merge(config.training.loss_and_metrics.default, dataset_config)
        return dataset_config

    # Fallback: use legacy structure (backwards compatibility)
    return OmegaConf.create(
        {"training_loss": config.training.training_loss, "validation_metrics": config.training.validation_metrics},
    )


def get_dataset_variable_groups(config: DictConfig, dataset_name: str | None = None) -> DictConfig:
    """Get dataset-specific variable groups configuration.

    Parameters
    ----------
    config : DictConfig
        The full configuration object
    dataset_name : str, optional
        Name of the dataset. If None, uses legacy structure (single-dataset mode).

    Returns
    -------
    DictConfig
        Dataset-specific variable groups configuration
    """
    # Try new loss_and_metrics structure first
    loss_metrics_config = get_dataset_loss_and_metrics_config(config, dataset_name)
    if hasattr(loss_metrics_config, "variable_groups"):
        return loss_metrics_config.variable_groups

    # Fallback to legacy structure
    if hasattr(config, "training") and hasattr(config.training, "variable_groups"):
        return config.training.variable_groups

    # Default fallback
    return OmegaConf.create({"default": "sfc"})


def get_dataset_metrics(config: DictConfig, dataset_name: str | None = None) -> list:
    """Get dataset-specific metrics configuration.

    Parameters
    ----------
    config : DictConfig
        The full configuration object
    dataset_name : str, optional
        Name of the dataset. If None, uses legacy structure (single-dataset mode).

    Returns
    -------
    list
        Dataset-specific metrics list
    """
    # Try new loss_and_metrics structure first
    loss_metrics_config = get_dataset_loss_and_metrics_config(config, dataset_name)
    if hasattr(loss_metrics_config, "metrics"):
        return OmegaConf.to_container(loss_metrics_config.metrics, resolve=True)

    # Fallback to legacy structure
    if hasattr(config, "training") and hasattr(config.training, "metrics"):
        return OmegaConf.to_container(config.training.metrics, resolve=True)

    # Default fallback
    return []


def is_multi_dataset_config(config: DictConfig) -> bool:
    """Check if the configuration is set up for multi-dataset mode.

    Parameters
    ----------
    config : DictConfig
        The full configuration object

    Returns
    -------
    bool
        True if multi-dataset configuration is detected
    """
    return hasattr(config, "data") and hasattr(config.data, "datasets") and len(config.data.datasets) > 0


def parse_multi_dataset_config(config: DictConfig) -> DictConfig:
    """Utility function to parse multi dataset config file with per_dataset_overrides.

    Inserts the override value if it exists or inserts the default value in the expected 
    format if not, and injects this into the config
    
    injected value:
    PER_DATASET_CONFIG_KEY.dataset_name.default_value or
    PER_DATASET_CONFIG_KEY.dataset_name.override_value

    """

    PER_DATASET_CONFIG_KEYS = [
        "data.dataset_specific",
        "dataloader.grid_indices",
        "dataloader.training",
        "dataloader.validation",
        "dataloader.test",
        "model.encoder",
        "model.decoder",
        "model.residual",
        "model.output_mask",
        "model.trainable_parameters.data",
        "model.bounding",
        "training.training_loss",
        "training.validation_metrics",
        "training.variable_groups",
        "training.metrics",
        "training.scalers",
        # TODO: Graphs - but its tricky with referencing to ${graphs.data} since this key will change
        # for multi datasets but be static data for single dataset.
    ]

    TRAINING_PERIOD_CONFIG_KEYS = [
        "dataloader.training",
        "dataloader.validation",
        "dataloader.test",
    ]

    datasets = list(config.get("datasets").keys()) #TODO: fix

    assert datasets == config.graph.data, f"{datasets}, {config.graph.data}"

    # TODO: consistency check, dataset names in overrides must all be in datasets

    ds_configs = DictConfig({})

    for dataset_name in datasets:

        # Generic options
        ds_configs[dataset_name] = DictConfig({})
        overrides = config.per_dataset_overrides.get(dataset_name, DictConfig({}))

        for key in PER_DATASET_CONFIG_KEYS:
            override = OmegaConf.select(overrides, key)
            default = OmegaConf.select(config, key)
            
            # Inject dataset from main config into dataloader config, TODO: need to improve this, but it looks
            # like this will require a larger change to the base config
            if key in TRAINING_PERIOD_CONFIG_KEYS:
                default.dataset = config.datasets[dataset_name]

            if override is not None:            
                override = merge_with_instantiate_override(default, override)
                #override = OmegaConf.merge(default, override)
            else:
                override = default
                
            _key = key + ".datasets." + dataset_name
            OmegaConf.update(ds_configs[dataset_name], _key, override, merge=True)
        
    # Clean up main config
    with open_dict(config):
        for key in PER_DATASET_CONFIG_KEYS:
            parent, child = key.rsplit(".", 1)
            value = OmegaConf.select(config, parent)
            if value and child in value:
                del value[child]
        # Insert dataset configs
        for ds in datasets:
            config = OmegaConf.merge(config, ds_configs[ds])
    
    return config

def is_instantiate_cfg(cfg):
    return (
        isinstance(cfg, DictConfig)
        and "_target_" in cfg
    )

def merge_with_instantiate_override(base, override):
    # If override is an instantiate config fully replace
    if is_instantiate_cfg(override):
        return override

    # If either is not a DictConfig use default OmegaConf behavior
    if not isinstance(base, DictConfig) or not isinstance(override, DictConfig):
        return override

    # Otherwise recurse key-by-key
    result = OmegaConf.create()

    for key in set(base.keys()) | set(override.keys()):
        if key in base and key in override:
            result[key] = merge_with_instantiate_override(
                base[key], override[key]
            )
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]

    return result
    
def get_trainable_parameters_config(cfg: dict) -> dict:
    from anemoi.utils.config import merge_configs

    return merge_configs(
        {"hidden": cfg.hidden},
        cfg.data.datasets
    )





        
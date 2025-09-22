"""Configuration utilities for handling dataset-specific configurations."""

from omegaconf import DictConfig
from omegaconf import OmegaConf


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
    if hasattr(config, "data") and hasattr(config.data, "datasets") and dataset_name in config.data.datasets:

        dataset_config = config.data.datasets[dataset_name]

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

# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from anemoi.training.utils.checkpoint_loaders import load_checkpoint_from_source

if TYPE_CHECKING:
    from pathlib import Path

    import torch

LOGGER = logging.getLogger(__name__)


class ModelLoader(ABC):
    """Abstract base class for loading models from checkpoints."""

    @abstractmethod
    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,
    ) -> torch.nn.Module:
        """Load model weights from checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            Target model to load weights into
        checkpoint_source : str | Path
            Checkpoint source (path, URL, etc.)
        strict : bool, optional
            Whether to strictly enforce that the keys in state_dict match
        **kwargs
            Additional arguments for specific loaders

        Returns
        -------
        torch.nn.Module
            Model with loaded weights
        """
        ...


class StandardModelLoader(ModelLoader):
    """Standard model loader for Lightning checkpoints."""

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> torch.nn.Module:
        """Load standard Lightning checkpoint weights."""
        checkpoint = load_checkpoint_from_source(checkpoint_source)

        if "state_dict" not in checkpoint:
            msg = f"No 'state_dict' found in checkpoint from {checkpoint_source}"
            raise ValueError(msg)

        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=strict)

        # Preserve checkpoint metadata for compatibility
        if hasattr(model, "_ckpt_model_name_to_index") and "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "data_indices" in hyper_params:
                model._ckpt_model_name_to_index = hyper_params["data_indices"].name_to_index

        LOGGER.info("Loaded model weights from %s", checkpoint_source)
        return model


class TransferLearningModelLoader(ModelLoader):
    """Model loader for transfer learning with size mismatch handling."""

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = False,
        skip_mismatched: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> torch.nn.Module:
        """Load weights with transfer learning compatibility."""
        checkpoint = load_checkpoint_from_source(checkpoint_source)

        if "state_dict" not in checkpoint:
            msg = f"No 'state_dict' found in checkpoint from {checkpoint_source}"
            raise ValueError(msg)

        state_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()

        if skip_mismatched:
            # Filter out layers with size mismatch
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        filtered_state_dict[key] = value
                    else:
                        LOGGER.info("Skipping parameter with size mismatch: %s", key)
                        LOGGER.info("Checkpoint shape: %s, Model shape: %s", value.shape, model_state_dict[key].shape)
                else:
                    LOGGER.info("Skipping parameter not in model: %s", key)

            state_dict = filtered_state_dict

        model.load_state_dict(state_dict, strict=strict)

        # Preserve checkpoint metadata for compatibility
        if hasattr(model, "_ckpt_model_name_to_index") and "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "data_indices" in hyper_params:
                model._ckpt_model_name_to_index = hyper_params["data_indices"].name_to_index

        model.weights_initialized = True
        LOGGER.info("Transfer learning applied successfully from %s", checkpoint_source)
        return model


class WeightsOnlyModelLoader(ModelLoader):
    """Model loader that only loads weights, not optimizer states or other metadata."""

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        strict: bool = True,
        **kwargs,
    ) -> torch.nn.Module:
        """Load only model weights from checkpoint."""
        # For weights-only loading, we use the TransferLearningModelLoader
        # as it has better handling of potential mismatches
        loader = TransferLearningModelLoader()
        return loader.load_model_weights(
            model=model,
            checkpoint_source=checkpoint_source,
            strict=strict,
            skip_mismatched=False,
            **kwargs,
        )


class ModelLoaderRegistry:
    """Registry for model loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, ModelLoader] = {}
        # Register default loaders
        self.register("standard", StandardModelLoader())
        self.register("transfer_learning", TransferLearningModelLoader())
        self.register("weights_only", WeightsOnlyModelLoader())

    def register(self, name: str, loader: ModelLoader) -> None:
        """Register a model loader."""
        self._loaders[name] = loader

    def get_loader(self, name: str) -> ModelLoader:
        """Get loader by name."""
        if name not in self._loaders:
            msg = f"Unknown loader: {name}. Available: {list(self._loaders.keys())}"
            raise ValueError(msg)
        return self._loaders[name]

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_source: str | Path,
        loader_type: str = "standard",
        **kwargs,
    ) -> torch.nn.Module:
        """Load model weights using specified loader."""
        loader = self.get_loader(loader_type)
        return loader.load_model_weights(model, checkpoint_source, **kwargs)


# Global registry instance
model_loader_registry = ModelLoaderRegistry()


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_source: str | Path,
    loader_type: str = "standard",
    **kwargs,
) -> torch.nn.Module:
    """Load model weights from checkpoint using specified loader.

    Parameters
    ----------
    model : torch.nn.Module
        Target model to load weights into
    checkpoint_source : str | Path
        Checkpoint source (local path, S3 URL, HTTP URL, etc.)
    loader_type : str, optional
        Type of loader to use ("standard", "transfer_learning", "weights_only")
    **kwargs
        Additional arguments passed to the loader

    Returns
    -------
    torch.nn.Module
        Model with loaded weights
    """
    return model_loader_registry.load_model_weights(
        model=model,
        checkpoint_source=checkpoint_source,
        loader_type=loader_type,
        **kwargs,
    )

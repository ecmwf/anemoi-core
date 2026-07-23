# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import os

import torch
from omegaconf import DictConfig

from anemoi.models.utils.compile import mark_for_compilation

LOGGER = logging.getLogger(__name__)


def subset_tensor(
    x: torch.Tensor,
    subset_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, int | None]:
    """Wrapper around torch.index_select to subset a tensor along a given dimension.

    'x_subset = x[subset_indices]' will likely not compile, but 'torch.index_select(x, dim, index)' will.
    This wrapper exists to support rewriting the subsetting operation in torch.compile()-friendly way.

    The subset indices can be a tuple of indices or a single index or None.
    The subset_indices might not be a tensor, in which case it will be converted to a tensor.

    tuple can also contain Ellipsis to indicate that the last dimension should be used for subsetting.
    These must be guarded explicitly because torch.index_select does not support Ellipsis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, int | None]
    The subsetted tensor, the subset indices, and the subset dimension
    """
    # Guard against Ellipsis and None, which are not supported by torch.index_select
    # e.g. subset_indices = (...,) or subset_indices = None
    # means that the last dimension should be used for subsetting
    if subset_indices is None or (len(subset_indices) == 1 and subset_indices[0] is Ellipsis):
        return x, None, None

    # Guard against Ellipsis in the subset indices,
    # which indicates that the last dimension should be used for subsetting
    # e.g. (..., indices) means that the last dimension should be used for subsetting
    if subset_indices[0] is Ellipsis:
        subset_dim = -1
        subset_index = subset_indices[-1]
    else:
        subset_dim = 0
        subset_index = subset_indices[0] if len(subset_indices) == 1 else subset_indices

    # Convert subset_index to a tensor if it is not already one, and move it to the same device as x
    if not isinstance(subset_index, torch.Tensor):
        subset_index = torch.as_tensor(
            subset_index,
            device=x.device,
            dtype=torch.long,
        )
    else:
        subset_index = subset_index.to(device=x.device, dtype=torch.long)

    # perform the subsetting using torch.index_select
    return torch.index_select(x, dim=subset_dim, index=subset_index), subset_index, subset_dim


def check_env_and_warn() -> None:
    """Reads env for settings which interfere with compilation and gives a warning.

    checks 'PYTORCH_CUDA_ALLOC_CONF' for 'expandable_segments:true', this can cause
    null pointer exceptions when compiling with certain versions of pytorch.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    # convert 'keyword1:value,keyword2:value,...' into a dict
    options = dict(item.split(":", 1) for item in conf.split(",") if ":" in item)

    # check if expandable segments is true
    using_expandable_segments = options.get("expandable_segments", "False").lower() == "true"
    if using_expandable_segments:
        LOGGER.warning(
            "You are using the 'expandable_segments' option for PyTorchs CUDA caching memory "
            "allocator, alongside torch.compile()."
            "This can cause null pointer exceptions at runtime 'RuntimeError: Expected "
            "curr_block->next == nullptr to be true, but got false.'"
            "To avoid this error, unset expandable segments e.g. 'unset PYTORCH_CUDA_ALLOC_CONF' "
            "or try upgrading your PyTorch.",
        )


def prepare_compilation(
    model: torch.nn.Module,
    model_config: DictConfig,
    training_config: DictConfig,
) -> torch.nn.Module:
    """Reads model_config and marks the matching submodules in model for compilation."""
    if hasattr(model_config, "compile"):
        model = mark_for_compilation(model, model_config.compile)
    recompile_limit = getattr(model_config, "recompile_limit", None)
    if hasattr(training_config, "recompile_limit"):
        LOGGER.warning(
            "The recompile_limit in config.training is deprecated. Please use config.model.recompile_limit instead.",
        )
        recompile_limit = getattr(training_config, "recompile_limit", None)
    if recompile_limit is not None:
        torch._dynamo.config.cache_size_limit = int(recompile_limit)
        torch._dynamo.config.accumulated_cache_size_limit = max(8 * int(recompile_limit), 256)
        LOGGER.info(
            "Recompile limit set to %d per kernel, %d accumulated",
            torch._dynamo.config.cache_size_limit,
            torch._dynamo.config.accumulated_cache_size_limit,
        )
    return model

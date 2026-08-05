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
from pathlib import Path

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
    options: dict[str, str] = {}
    for item in conf.split(","):
        if ":" not in item:
            continue
        key, value = (s.strip() for s in item.split(":", 1))
        options[key] = value

    # check if expandable segments is true
    using_expandable_segments = options.get("expandable_segments", "False").lower() == "true"
    if using_expandable_segments:
        LOGGER.warning(
            "You are using the 'expandable_segments' option for PyTorch's CUDA caching memory "
            "allocator, alongside torch.compile(). "
            "This can cause null pointer exceptions at runtime: 'RuntimeError: Expected "
            "curr_block->next == nullptr to be true, but got false.' "
            "To avoid this error, unset expandable segments (e.g. 'unset PYTORCH_CUDA_ALLOC_CONF') "
            "or try upgrading your PyTorch.",
        )


def prepare_compilation(
    model: torch.nn.Module,
    model_config: DictConfig,
    training_config: DictConfig,
) -> torch.nn.Module:
    """Reads model_config and marks the matching submodules in model for compilation."""
    torch.set_num_threads(1)  # avoid thread count changing and triggering needless recompilation
    # allow torch compile to pass logging calls through to the logger, otherwise it will try to compile them and error
    torch._dynamo.config.ignore_logger_methods.add(logging.Logger.info)
    torch._dynamo.config.ignore_logger_methods.add(logging.Logger.warning)
    if hasattr(model_config, "compile"):
        model = mark_for_compilation(model, model_config.compile)
        check_env_and_warn()  # warn if env settings interfere with compilation
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


def load_compile_cache(compile_cache_file: str) -> None:
    """Load the torch.compile cache from disk if it exists.

    If found, it will prepopulate all the relevant caches:
        FXGraphCache: A cache of graph-based IR components used in compilation.
        TritonCache: A cache of Triton-compilation results, including cubin files
             generated by Triton and other caching artifacts.
        InductorCache: A bundle of FXGraphCache and Triton cache.
        AOTAutogradCache: A cache of joint graph artifacts.
        PGO-cache: A cache of dynamic shape decisions to reduce number of recompilations.
        AutotuningCache: results of benchmarking different inductor-generated triton kernels.

    By default, these caches are stored under 'TORCHINDUCTOR_CACHE_DIR'.
    """
    # TMPDIR must be the same name across all ranks and jobs..can't use TMPDIR if it has a process pid in it
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ.get("TMPDIR") + "/anemoi_compile_cache"
    os.environ["TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE"] = (
        "0"  # disable local cache for autotuning, since it contains hardcoded paths which
        # breaks when loading the cache on a different node
    )
    # only local rank should load the cache, since it is shared across all ranks on a node
    gpus_per_node = 4  # TODO(cathal): get this from config
    # TODO(cathal): torch distributed is not initialized at this point (called from training/train.py)
    if torch.distributed.is_initialized() and torch.distributed.get_rank() % gpus_per_node != 0:
        return

    if compile_cache_file is None:
        return

    path = Path(compile_cache_file)
    if not path.exists():
        LOGGER.info("No torch.compile cache found at %s", compile_cache_file)
        return
    if not path.is_file():
        LOGGER.warning("torch cache byte-file '%s' is not a file", compile_cache_file)
        return

    LOGGER.info("Loading torch.compile cache from %s", compile_cache_file)
    artifact_bytes = path.read_bytes()
    maybe_cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
    if maybe_cache_info is not None:
        LOGGER.debug("Loaded torch.compile cache info: %s", maybe_cache_info)


def save_compile_cache(compile_cache_file: str) -> None:
    """Save the torch.compile cache to disk in bytes."""
    # only rank 0 saves the cache
    # TODO(cathal): you might want one rank per node to do this if you had many shapes

    # TODO(cathal): seems like the code will be restored to the original location
    # if this has a process pid e.g. the default, this will lead to perm denied

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    if compile_cache_file is None:
        return

    path = Path(compile_cache_file)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving torch.compile cache to %s", compile_cache_file)
    artifacts = torch.compiler.save_cache_artifacts()

    if artifacts is None:
        LOGGER.info("No torch.compile cache artifacts to save.")
        return
    artifact_bytes, cache_info = artifacts

    path.write_bytes(artifact_bytes)
    LOGGER.info("Torch compile cache saved to %s.", compile_cache_file)
    LOGGER.debug("Torch compile cache info: %s", cache_info)

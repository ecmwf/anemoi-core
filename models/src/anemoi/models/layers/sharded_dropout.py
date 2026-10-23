# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Refuse dropout when the model is sharded across GPUs."""

from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

# Every dropout layer torch ships (all subclasses of the private _DropoutNd).
_DROPOUT_MODULES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def _uses_dropout(module: nn.Module) -> bool:
    """Whether ``module`` drops anything: a dropout layer, or a ``dropout_p`` (attention), above zero."""
    for submodule in module.modules():
        if isinstance(submodule, _DROPOUT_MODULES) and submodule.p > 0:
            return True
        dropout_p = getattr(submodule, "dropout_p", None)
        if isinstance(dropout_p, int | float) and dropout_p > 0:
            return True
    return False


def refuse_dropout_when_sharded(module: nn.Module, model_comm_group: ProcessGroup | None) -> None:
    """Refuse dropout when the model is sharded across GPUs.

    Ranks in a model group share a random seed, so dropout on sharded data would draw
    correlated masks rather than independent ones. Every processor and mapper
    ``forward`` calls this first (a test checks that each one does). It counts dropout
    layers and any ``dropout_p`` above zero, including dropout added after
    construction, such as a LoRA adapter's. Only training is refused: dropout does
    nothing in eval mode, so sharded inference keeps working.

    Parameters
    ----------
    module : nn.Module
        The processor or mapper about to run.
    model_comm_group : ProcessGroup, optional
        The model communication group; ``None`` or size 1 means not sharded.

    Raises
    ------
    ValueError
        If the module is training, the model is sharded, and the module uses dropout.
    """
    # The group is only asked for its size when there is dropout to refuse.
    if not module.training or model_comm_group is None or not _uses_dropout(module):
        return
    if model_comm_group.size() > 1:
        raise ValueError(f"Dropout is not supported when the model is sharded across {model_comm_group.size()} GPUs")

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Refuse dropout when the model is sharded across GPUs."""

import inspect

from torch import nn

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


class ShardedDropoutGuard:
    """Refuse dropout when the model is sharded across GPUs.

    Ranks in a model group share a random seed, so dropout on sharded data would draw
    correlated masks rather than independent ones. Only training is refused: dropout
    does nothing in eval mode. Mixed into ``BaseProcessor`` and ``BaseMapper`` ahead of
    ``nn.Module``: the check runs on every call, before ``forward``, so a subclass
    cannot skip it, and it counts dropout added after construction (a LoRA adapter's,
    for instance).

    It overrides ``__call__`` rather than registering a forward pre-hook. A hook is
    stored on the module and pickled with it, which would tie every pickled inference
    checkpoint to the hook's import path; a method on the class is not pickled.
    """

    def __call__(self, *args, **kwargs):
        model_comm_group = kwargs.get("model_comm_group")
        if model_comm_group is None and "model_comm_group" not in kwargs:
            try:
                bound = inspect.signature(self.forward).bind_partial(*args, **kwargs)
            except TypeError:
                bound = None
            model_comm_group = bound.arguments.get("model_comm_group") if bound else None
        # Dropout only acts in training mode, so sharded inference is never refused.
        if self.training and model_comm_group is not None and model_comm_group.size() > 1 and _uses_dropout(self):
            raise ValueError(
                f"Dropout is not supported when the model is sharded across {model_comm_group.size()} GPUs"
            )
        return super().__call__(*args, **kwargs)

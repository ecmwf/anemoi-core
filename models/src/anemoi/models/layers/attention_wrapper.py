# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Base class of the attention backends of :class:`anemoi.models.layers.attention.MultiHeadSelfAttention`."""

from torch import nn


class AttentionWrapper(nn.Module):
    """Base class of the attention backends.

    Each backend lists in ``unsupported`` the options of MultiHeadSelfAttention it cannot honour,
    named as in the model config, and says in :meth:`check_head_dim` which head sizes it works
    with. MultiHeadSelfAttention checks both once, when it sets up the backend, so a config asking
    for something the backend cannot do fails when the model is built rather than at the first step.
    """

    unsupported: frozenset[str] = frozenset()

    def describe(self) -> str:
        """The backend's name in error messages."""
        return type(self).__name__

    def check_supported(self, **options: bool) -> None:
        """Raises if any of the options that are switched on is one this backend cannot honour."""
        requested = sorted(name for name, used in options.items() if used and name in self.unsupported)
        if requested:
            raise NotImplementedError(
                f"{self.describe()} does not support: {', '.join(requested)}. "
                "Please switch to a different attention_implementation, or disable these options."
            )

    def check_head_dim(self, head_dim: int) -> None:
        """Raises if the backend cannot work with heads of ``head_dim`` channels; any size works by default."""

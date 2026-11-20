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

    ``unsupported`` lists the attention options a backend cannot handle, named as in the model
    config, and :meth:`check_head_dim` the head sizes it rejects. Both are checked when the model is
    built, so a bad config fails early.
    """

    unsupported: frozenset[str] = frozenset()

    def describe(self) -> str:
        """The backend's name in error messages."""
        return type(self).__name__

    def check_supported(self, **options: bool) -> None:
        """Raise if an option that is switched on is unsupported."""
        requested = sorted(name for name, used in options.items() if used and name in self.unsupported)
        if requested:
            raise NotImplementedError(
                f"{self.describe()} does not support: {', '.join(requested)}. "
                "Please switch to a different attention_implementation, or disable these options."
            )

    def check_head_dim(self, head_dim: int) -> None:
        """Raise if the backend cannot use heads of ``head_dim`` channels; any size works by default."""

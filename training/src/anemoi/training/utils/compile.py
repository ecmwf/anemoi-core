# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import torch


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
        subset_index = subset_indices

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

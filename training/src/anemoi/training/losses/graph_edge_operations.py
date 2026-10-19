# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared helpers for graph-edge scores."""

import torch

from anemoi.training.losses.graph_score_base import safe_sqrt


def edge_difference(
    node_values: torch.Tensor,
    source_index: torch.Tensor,
    destination_index: torch.Tensor,
) -> torch.Tensor:
    """Return ``source - destination`` for every graph edge."""
    return node_values[..., source_index, :] - node_values[..., destination_index, :]


def compute_node_validity(
    y_pred_ens: torch.Tensor,
    y_target: torch.Tensor,
    *,
    ignore_nans: bool,
) -> torch.Tensor | None:
    """Return nodes that are finite in the target and every ensemble member."""
    if not ignore_nans:
        return None
    return torch.isfinite(y_target) & torch.isfinite(y_pred_ens).all(dim=2)


def compute_edge_validity(
    node_valid: torch.Tensor | None,
    source_index: torch.Tensor,
    destination_index: torch.Tensor,
    edge_weights: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute edge validity and valid weight sums from a node-validity mask."""
    if node_valid is None:
        return None, None

    edge_valid = node_valid[..., source_index, :] & node_valid[..., destination_index, :]
    weight_shape = (1,) * (edge_valid.ndim - 2) + (-1, 1)
    valid_edge_weights = edge_valid.to(dtype=edge_weights.dtype) * edge_weights.view(weight_shape)
    valid_weight_sum = torch.zeros_like(node_valid, dtype=edge_weights.dtype)
    valid_weight_sum.index_add_(-2, destination_index, valid_edge_weights)
    return edge_valid, valid_weight_sum


def aggregate_edge_values(
    edge_values: torch.Tensor,
    destination_index: torch.Tensor,
    edge_weights: torch.Tensor,
    num_nodes: int,
    *,
    node_valid: torch.Tensor | None,
    edge_valid: torch.Tensor | None,
    valid_weight_sum: torch.Tensor | None,
    row_normalize: bool,
) -> torch.Tensor:
    """Return a weighted sum for each graph row with missing-value normalization."""
    weight_shape = (1,) * (edge_values.ndim - 2) + (-1, 1)
    weights = edge_weights.to(dtype=edge_values.dtype).view(weight_shape)
    if edge_valid is not None:
        edge_values = torch.where(edge_valid, edge_values, torch.zeros_like(edge_values))

    node_values = torch.zeros(
        (*edge_values.shape[:-2], num_nodes, edge_values.shape[-1]),
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    node_values.index_add_(-2, destination_index, edge_values * weights)

    if node_valid is None:
        return node_values

    assert valid_weight_sum is not None
    if row_normalize:
        safe_weight_sum = torch.where(
            valid_weight_sum > 0,
            valid_weight_sum,
            torch.ones_like(valid_weight_sum),
        )
        node_values = node_values / safe_weight_sum
    return node_values.masked_fill((valid_weight_sum <= 0) | ~node_valid, torch.nan)


def weighted_row_l2_norm(
    edge_values: torch.Tensor,
    destination_index: torch.Tensor,
    edge_weights: torch.Tensor,
    num_nodes: int,
    *,
    node_valid: torch.Tensor | None,
    edge_valid: torch.Tensor | None,
    valid_weight_sum: torch.Tensor | None,
    row_normalize: bool,
) -> torch.Tensor:
    """Return a numerically stable weighted L2 norm for every graph row."""
    input_shape = edge_values.shape
    flat_edge_values = edge_values.reshape(-1, *input_shape[-2:])
    flat_batch_size, _, num_variables = flat_edge_values.shape
    row_indices = destination_index.view(1, -1, 1).expand(flat_batch_size, -1, num_variables)
    weight_view = edge_weights.to(dtype=edge_values.dtype).view(1, -1, 1)

    active_edges = (weight_view > 0).expand_as(flat_edge_values)
    if edge_valid is not None:
        flat_valid = edge_valid.reshape_as(flat_edge_values)
        active_edges = active_edges & flat_valid

    safe_edge_values = torch.where(
        active_edges,
        flat_edge_values,
        torch.zeros_like(flat_edge_values),
    )
    gathered_abs = torch.abs(safe_edge_values)
    row_max = torch.zeros(
        flat_batch_size,
        num_nodes,
        num_variables,
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    row_max.scatter_reduce_(
        1,
        row_indices,
        gathered_abs,
        reduce="amax",
        include_self=False,
    )
    # The scale cancels from the norm, so its derivative is unnecessary.
    row_max = row_max.detach()

    gathered_row_max = row_max[:, destination_index, :]
    safe_row_max = torch.where(
        gathered_row_max > 0,
        gathered_row_max,
        torch.ones_like(gathered_row_max),
    )
    scaled_abs = torch.where(
        gathered_row_max > 0,
        gathered_abs / safe_row_max,
        torch.zeros_like(gathered_abs),
    )

    effective_weights = torch.where(active_edges, weight_view, torch.zeros_like(weight_view))
    if edge_valid is not None and row_normalize:
        assert valid_weight_sum is not None
        flat_valid_weight_sum = valid_weight_sum.reshape(flat_batch_size, num_nodes, num_variables)
        gathered_weight_sum = flat_valid_weight_sum[:, destination_index, :]
        safe_weight_sum = torch.where(
            gathered_weight_sum > 0,
            gathered_weight_sum,
            torch.ones_like(gathered_weight_sum),
        )
        effective_weights = effective_weights / safe_weight_sum

    squared_norm = torch.zeros_like(row_max)
    squared_norm.index_add_(1, destination_index, effective_weights * scaled_abs.square())
    row_norm = row_max * safe_sqrt(squared_norm)

    if node_valid is not None:
        assert valid_weight_sum is not None
        flat_node_valid = node_valid.reshape(flat_batch_size, num_nodes, num_variables)
        flat_valid_weight_sum = valid_weight_sum.reshape(flat_batch_size, num_nodes, num_variables)
        row_norm = row_norm.masked_fill((flat_valid_weight_sum <= 0) | ~flat_node_valid, torch.nan)

    return row_norm.reshape(*input_shape[:-2], num_nodes, num_variables)


class _WeightedEdgeRowL2Norm(torch.autograd.Function):
    """Evaluate stable edge-row norms without retaining edge-sized tensors."""

    @staticmethod
    def forward(
        ctx,  # noqa: ANN001
        node_values: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_weights: torch.Tensor,
        node_valid: torch.Tensor | None,
        edge_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
        row_normalize: bool,
    ) -> torch.Tensor:
        row_norm = weighted_row_l2_norm(
            edge_difference(node_values, source_index, destination_index),
            destination_index,
            edge_weights,
            node_values.shape[-2],
            node_valid=node_valid,
            edge_valid=edge_valid,
            valid_weight_sum=valid_weight_sum,
            row_normalize=row_normalize,
        )
        ctx.save_for_backward(
            node_values,
            source_index,
            destination_index,
            edge_weights,
            row_norm,
            edge_valid,
            valid_weight_sum,
        )
        ctx.row_normalize = row_normalize
        return row_norm

    @staticmethod
    def backward(
        ctx,  # noqa: ANN001
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        node_values, source_index, destination_index, edge_weights, row_norm, edge_valid, valid_weight_sum = (
            ctx.saved_tensors
        )
        edge_values = edge_difference(node_values, source_index, destination_index)
        gathered_norm = row_norm[..., destination_index, :]
        positive_norm = gathered_norm > 0
        weight_view = edge_weights.to(dtype=node_values.dtype).view((1,) * (node_values.ndim - 2) + (-1, 1))
        active_edges = positive_norm & (weight_view > 0)
        if edge_valid is not None:
            active_edges = active_edges & edge_valid

        safe_norm = torch.where(positive_norm, gathered_norm, torch.ones_like(gathered_norm))
        edge_gradient = grad_output[..., destination_index, :] * weight_view
        edge_gradient = edge_gradient * (edge_values / safe_norm)
        if edge_valid is not None and ctx.row_normalize:
            assert valid_weight_sum is not None
            gathered_weight_sum = valid_weight_sum[..., destination_index, :]
            safe_weight_sum = torch.where(
                gathered_weight_sum > 0,
                gathered_weight_sum,
                torch.ones_like(gathered_weight_sum),
            )
            edge_gradient = edge_gradient / safe_weight_sum
        edge_gradient = torch.where(active_edges, edge_gradient, torch.zeros_like(edge_gradient))

        node_gradient = torch.zeros_like(node_values)
        node_gradient.index_add_(-2, source_index, edge_gradient)
        node_gradient.index_add_(-2, destination_index, -edge_gradient)
        return node_gradient, None, None, None, None, None, None, None


def weighted_edge_row_l2_norm(
    node_values: torch.Tensor,
    source_index: torch.Tensor,
    destination_index: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    node_valid: torch.Tensor | None,
    edge_valid: torch.Tensor | None,
    valid_weight_sum: torch.Tensor | None,
    row_normalize: bool,
) -> torch.Tensor:
    """Return stable edge-row norms with a memory-efficient analytical backward."""
    return _WeightedEdgeRowL2Norm.apply(
        node_values,
        source_index,
        destination_index,
        edge_weights,
        node_valid,
        edge_valid,
        valid_weight_sum,
        row_normalize,
    )

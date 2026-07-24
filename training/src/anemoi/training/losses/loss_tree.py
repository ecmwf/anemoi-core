# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Loss trees and their reduction helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LossTree:
    """A named, weighted leaf or group in a loss tree.

    A leaf contains ``value`` and no children. A group contains one or more
    children and no value. The tree retains the structure of composed losses
    until a caller selects the reduction needed for training or diagnostics.
    """

    name: str
    weight: float = 1.0
    value: torch.Tensor | None = None
    children: tuple[LossTree, ...] = ()

    def __post_init__(self) -> None:
        # Every node represents exactly one leaf value or one group of children.
        if self.value is None and not self.children:
            msg = "LossTree must contain a value or at least one child."
            raise ValueError(msg)

        if self.value is not None and self.children:
            msg = "LossTree cannot contain both a value and children."
            raise ValueError(msg)


def as_loss_tree(
    loss: torch.Tensor | LossTree,
    *,
    name: str,
    weight: float = 1.0,
) -> LossTree:
    """Wrap a tensor or loss tree in a named, weighted output node."""
    if isinstance(loss, torch.Tensor):
        # A regular loss tensor becomes a leaf in the tree.
        return LossTree(name=name, weight=weight, value=loss)
    if isinstance(loss, LossTree):
        # Keep a nested loss tree intact and apply the new weight to the group.
        return LossTree(name=name, weight=weight, children=(loss,))
    msg = f"Expected a loss tensor or LossTree, got {type(loss).__name__}."
    raise TypeError(msg)


def _collect_loss_components(
    loss: LossTree,
    components: dict[str, torch.Tensor],
    *,
    path: str,
    parent_weight: float,
) -> None:
    """Collect weighted leaves while retaining their path through the tree."""
    weight = parent_weight * loss.weight

    if loss.value is not None:
        components[path] = weight * loss.value
        return

    for child in loss.children:
        child_path = f"{path}/{child.name}" if path else child.name
        _collect_loss_components(
            child,
            components,
            path=child_path,
            parent_weight=weight,
        )


def loss_components(loss: torch.Tensor | LossTree) -> dict[str, torch.Tensor]:
    """Return weighted loss leaves by name for diagnostic logging.

    Plain tensors have no name of their own and therefore use an empty key.
    Structured losses retain the path of each leaf through the loss tree.
    """
    if isinstance(loss, torch.Tensor):
        return {"": loss}

    components: dict[str, torch.Tensor] = {}
    if loss.value is not None:
        _collect_loss_components(loss, components, path=loss.name, parent_weight=1.0)
        return components

    # The root identifies the complete loss. Diagnostic names start at its
    # children, which are the independently useful values to log.
    for child in loss.children:
        _collect_loss_components(
            child,
            components,
            path=child.name,
            parent_weight=loss.weight,
        )
    return components


def sum_loss(loss: torch.Tensor | LossTree) -> torch.Tensor:
    """Add all weighted loss values and return one scalar for training."""
    if isinstance(loss, torch.Tensor):
        # Every tensor value contributes to the training loss.
        return loss.sum()

    if loss.value is not None:
        # Reduce each leaf before applying its weight
        return loss.weight * loss.value.sum()

    # Reduce each child independently before combining their totals.
    child_losses = [sum_loss(child) for child in loss.children]
    if len(child_losses) == 1:
        return loss.weight * child_losses[0]

    return loss.weight * torch.stack(child_losses).sum()


def loss_per_variable(loss: torch.Tensor | LossTree) -> dict[str, torch.Tensor]:
    """Reduce each weighted loss leaf while retaining its variable dimension."""
    per_variable = {}
    for name, component in loss_components(loss).items():
        if component.ndim > 1:
            # Sum the leading dimensions and retain the final variable dimension.
            leading_dimensions = tuple(range(component.ndim - 1))
            component = component.sum(dim=leading_dimensions)
        per_variable[name] = component
    return per_variable


def sum_loss_per_variable(
    loss: torch.Tensor | LossTree,
    *,
    num_variables: int,
) -> torch.Tensor | None:
    """Combine the loss leaves that contain one value per output variable.

    Only components matching ``num_variables`` are included. Returns ``None``
    when no component matches.
    """
    component_losses = loss_per_variable(loss)

    losses_to_sum = []
    ignored_components = []
    for name, component in component_losses.items():
        if component.ndim == 0 and num_variables == 1:
            component = component.unsqueeze(0)

        if component.ndim == 1 and component.shape[0] == num_variables:
            losses_to_sum.append(component)
        else:
            ignored_components.append(name or "unnamed")

    if ignored_components:
        LOGGER.warning(
            "Ignoring loss components without one value per output variable: %s",
            ", ".join(ignored_components),
        )

    if not losses_to_sum:
        return None
    if len(losses_to_sum) == 1:
        return losses_to_sum[0]

    return torch.stack(losses_to_sum).sum(dim=0)

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import torch
from torch.distributed.distributed_c10d import ProcessGroup


@runtime_checkable
class ModelInterface(Protocol):
    """Interface for Anemoi models."""

    data_nodes_name: str | list[str]
    "Name(s) of data nodes in the graph. The model declares which graph nodes it uses."

    def pre_process(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply pre-processing (e.g. normalisation) to model inputs."""
        ...

    def post_process(self, y: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply post-processing (e.g. denormalisation) to model outputs."""
        ...

    def forward(
        self,
        x: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the model forward pass on pre-processed inputs."""
        ...

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run a full inference step (pre-process → forward → post-process)."""
        ...


@runtime_checkable
class DiffusionModelInterface(ModelInterface, Protocol):
    """Interface for models that support diffusion tasks."""

    def get_diffusion_parameters(self) -> tuple[float, float, float]:
        """Return ``(sigma_max, sigma_min, sigma_data)`` for diffusion training."""
        ...

    def forward_with_preconditioning(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the diffusion forward pass with model-specific preconditioning."""
        ...

    def apply_imputer_inverse(self, dataset_name: str, x: torch.Tensor) -> torch.Tensor:
        """Map output-space tensors back through any inverse imputation logic."""
        ...

    def apply_reference_state_truncation(
        self,
        x: dict[str, torch.Tensor],
        grid_shard_shapes,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare reference states used by diffusion tasks."""
        ...


@runtime_checkable
class DiffusionTendencyModelInterface(DiffusionModelInterface, Protocol):
    """Interface for diffusion models that predict tendencies."""

    def get_tendency_processors(self, dataset_name: str) -> tuple[object, object]:
        """Return the pre/post tendency processors for one dataset."""
        ...

    def compute_tendency_step(
        self,
        dataset_name: str,
        y_step: torch.Tensor,
        x_ref_step: torch.Tensor,
        tendency_pre_processor: object,
    ) -> torch.Tensor:
        """Convert one output step into a tendency target."""
        ...

    def add_tendency_to_state_step(
        self,
        dataset_name: str,
        x_ref_step: torch.Tensor,
        tendency_step: torch.Tensor,
        tendency_post_processor: object,
    ) -> torch.Tensor:
        """Reconstruct one state step from a tendency prediction."""
        ...


__all__ = [
    "ModelInterface",
    "DiffusionModelInterface",
    "DiffusionTendencyModelInterface",
]

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Export runtime encoder and decoder mapper graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import torch
from pytorch_lightning.callbacks import Callback

from anemoi.models.layers.graph_provider import DynamicGraphProvider

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class MapperGraphExport(Callback):
    """Capture and save dynamic mapper graphs for one selected training batch."""

    def __init__(self, output_dir: str | Path, batch_index: int = 0) -> None:
        super().__init__()
        if batch_index < 0:
            message = "batch_index must be non-negative."
            raise ValueError(message)
        self.output_dir = Path(output_dir)
        self.batch_index = batch_index
        self._armed_providers: list[tuple[str, str, DynamicGraphProvider]] = []
        self._completed = False

    def state_dict(self) -> dict[str, bool]:
        """Persist one-shot completion across checkpoint resumes."""
        return {"completed": self._completed}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore one-shot completion from a Lightning checkpoint."""
        self._completed = bool(state_dict.get("completed", False))

    @staticmethod
    def _torch_drop_down(pl_module: pl.LightningModule) -> torch.nn.Module:
        model = pl_module.model
        return model.module.model if hasattr(model, "module") else model.model

    @staticmethod
    def _summary(graph: HeteroData) -> str:
        edge_name = graph.edge_types[0]
        source_name, _, target_name = edge_name
        edge_index = graph[edge_name].edge_index
        source_count = graph[source_name].num_nodes
        target_count = graph[target_name].num_nodes
        source_degree = torch.bincount(edge_index[0], minlength=source_count)
        target_degree = torch.bincount(edge_index[1], minlength=target_count)
        return (
            f"nodes=({source_name}:{source_count}, {target_name}:{target_count}), "
            f"edges={edge_index.shape[1]}, "
            f"source_degree=min:{source_degree.min().item() if source_count else 0},"
            f"max:{source_degree.max().item() if source_count else 0},"
            f"isolated:{(source_degree == 0).sum().item()}, "
            f"target_degree=min:{target_degree.min().item() if target_count else 0},"
            f"max:{target_degree.max().item() if target_count else 0},"
            f"isolated:{(target_degree == 0).sum().item()}"
        )

    def _arm_provider(
        self,
        provider: Any,
        dataset_name: str,
        role: str,
        source_name: str,
        target_name: str,
    ) -> None:
        if not isinstance(provider, DynamicGraphProvider):
            return
        provider.capture_next_graph(source_name, target_name)
        self._armed_providers.append((dataset_name, role, provider))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        if self._completed or batch_idx != self.batch_index or not trainer.is_global_zero:
            return

        model = self._torch_drop_down(pl_module)
        hidden_name = model._graph_name_hidden
        for dataset_name, provider in model.encoder_graph_provider.items():
            self._arm_provider(
                provider,
                dataset_name,
                "encoder",
                f"{dataset_name}_input",
                hidden_name,
            )
        for dataset_name, provider in model.decoder_graph_provider.items():
            self._arm_provider(
                provider,
                dataset_name,
                "decoder",
                hidden_name,
                f"{dataset_name}_output",
            )

        if not self._armed_providers:
            LOGGER.warning("MapperGraphExport found no dynamic encoder or decoder graph providers.")
            self._completed = True

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        if self._completed or batch_idx != self.batch_index or not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for dataset_name, role, provider in self._armed_providers:
            graph = provider.consume_captured_graph()
            if graph is None:
                missing.append(f"{dataset_name}:{role}")
                continue

            path = self.output_dir / f"batch_{batch_idx:06d}_{dataset_name}_{role}.pt"
            torch.save(graph, path)
            LOGGER.info("Saved mapper graph to %s (%s).", path, self._summary(graph))

        self._armed_providers.clear()
        self._completed = True
        if missing:
            LOGGER.warning(
                "Mapper graph capture did not run for inactive providers: %s.",
                ", ".join(missing),
            )

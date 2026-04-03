# (C) Copyright 2024 Anemoi contributors.
# Copyright (C) Bull S.A.S - 2025
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model

from anemoi.training.train.tasks import GraphForecaster

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema


LOGGER = logging.getLogger(__name__)


class LoRAGraphForecaster(GraphForecaster):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.lora_config = LoraConfig(**config.model_dump(by_alias=True).training.lora_config)

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'LoRAGraphForecaster' in checkpoint['hyper_parameters']['config'].training.model_task:
            self._inject_lora_adapters()
    
    def on_checkpoint_loaded(self) -> None:
        if not isinstance(self.model, PeftModel):
            self._inject_lora_adapters()

    def _inject_lora_adapters(self) -> None:
        get_peft_model(self.model, self.lora_config)
        LOGGER.info("LoRA adapters injected into the model")

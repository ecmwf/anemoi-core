from collections.abc import Generator

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_structure.sample_provider import StaticMetadata
from anemoi.models.data_structure.sample_provider import merge_offsets
from anemoi.models.data_structure.structure import TreeDict
from anemoi.training.train.tasks.refactor.base import BaseGraphPLModule


class ForecastingPLModule(BaseGraphPLModule):
    def get_input_from_batch(self, metadata: StaticMetadata, batch):
        assert isinstance(metadata, StaticMetadata), type(metadata)
        new = TreeDict()
        for k, v in metadata.stack_offsets["input"].items():
            new[k] = merge_offsets(batch["input"][k])
        return new

    def get_target_from_batch(self, metadata: StaticMetadata, batch):
        assert isinstance(metadata, StaticMetadata), type(metadata)
        new = TreeDict()
        for k, v in metadata.stack_offsets["target"].items():
            new[k] = merge_offsets(batch["target"][k])
        return new

    def get_input_metadata_from_metadata(self, static_metadata: StaticMetadata) -> TreeDict:
        return self.get_input_from_batch(static_metadata, static_metadata.batch)

    def get_target_metadata_from_metadata(self, static_metadata: StaticMetadata) -> TreeDict:
        return self.get_target_from_batch(static_metadata, static_metadata.batch)

    def get_output_metadata_from_metadata(self, static_metadata: StaticMetadata) -> TreeDict:
        return self.get_target_from_batch(static_metadata, static_metadata.batch)

    def _step(
        self,
        batch,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        # ✅ here, batch = input + target
        # validation vs training, do we have validation batch or training batch?
        #
        print("️⚠️💬 Starting _step")
        batch = self.static_metadata.batch + batch

        batch = self.apply_normaliser_to_batch(batch)

        # get input and target
        input = self.get_input_from_batch(self.static_metadata, batch)
        target = self.get_target_from_batch(self.static_metadata, batch)
        print(input.to_str("⚠️input data"))
        print(target.to_str("⚠️target data"))

        loss = torch.zeros(1, dtype=target.first["data"].dtype, device=self.device, requires_grad=True)
        print(self.loss.to_str("⚠️loss function"))

        # graph = self.graph_editor.update_graph(self.graph_data, input_latlons, target_latlons)

        # run model for one step
        y_pred = self(input, self.graph_data.clone().to("cpu"))

        assert target.matches_keys(y_pred), (target.keys(), y_pred.keys)

        print(y_pred.to_str("⚠️y_pred after merging semantic info from target"))
        loss = 0.0
        for key, target_data in target.items():
            loss += checkpoint(self.loss[key], y_pred[key], target_data, use_reentrant=False)
        # loss *= 1 / len(batch["target"]) # Do we want to average over the number of targets??
        print("computed loss:", loss)

        metrics_next = {}
        if validation_mode:
            print("Validation metrics SKIPPED !!!")
            # metrics_next = self.calculate_val_metrics(y_pred, batch["target"], rollout_step=0)

        print(f"computed loss: {loss}, metrics: {metrics_next}, y_pred: {y_pred.to_str('y_pred')}")
        print("️⚠️💬 End of _step")
        return loss, metrics_next, y_pred

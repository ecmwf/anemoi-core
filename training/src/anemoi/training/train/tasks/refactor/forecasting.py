from collections.abc import Generator

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.refactor.base import BaseGraphPLModule


class ForecastingPLModule(BaseGraphPLModule):
    def get_input_from_batch(self, batch, **kwargs):
        return batch["input"]

    def get_target_from_batch(self, batch, **kwargs):
        return batch["target"]

    def get_semantic_from_metadata(self, target_metadata, target, **kwargs):
        semantic = target_metadata.new_empty()
        for k, v in target_metadata.items():
            box = v.copy()
            if "data" in box:
                box.pop("data")
            # allows to look for some information in the target
            if "latitudes" not in box and "latitudes" in target[k]:
                box["latitudes"] = target[k]["latitudes"]
            if "longitudes" not in box and "longitudes" in target[k]:
                box["longitudes"] = target[k]["longitudes"]
            if "timedeltas" not in box and "timedeltas" in target[k]:
                box["timedeltas"] = target[k]["timedeltas"]
            if "reference_date" in target[k]:
                box["reference_date"] = target[k]["reference_date"]
            if "reference_date_str" in target[k]:
                box["reference_date_str"] = target[k]["reference_date_str"]
            semantic[k] = box
        return semantic

    def _step(
        self,
        batch,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        # ✅ here, batch = input + target
        # validation vs training, do we have validation batch or training batch?
        #
        print("️⚠️💬 Starting _step")
        batch = self.batch_metadata + batch

        batch = self.apply_normaliser_to_batch(batch)

        loss = torch.zeros(1, dtype=batch.first["data"].dtype, device=self.device, requires_grad=True)
        print(self.loss.to_str("⚠️loss function"))

        # get input and target
        input = self.get_input_from_batch(batch)
        target = self.get_target_from_batch(batch)
        print(input.to_str("⚠️input data"))
        print(target.to_str("⚠️target data"))

        semantic = self.get_semantic_from_metadata(self.target_metadata, target)
        print(semantic.to_str("⚠️semantic info from target"))

        # graph = self.graph_editor.update_graph(self.graph_data, input_latlons, target_latlons)

        # run model for one step
        y_pred = self(input, self.graph_data.clone().to("cpu"))

        # y_pred = target.select_content(["data"])  # for development, don't keep this line
        print(y_pred.to_str("⚠️y_pred before merging semantic info from target"))

        # y_pred = semantic + y_pred
        # new_y = semantic.new_empty()
        # for k, v in semantic.items():
        #    box = v.copy()
        #    if isinstance(y_pred[k], torch.Tensor):
        #        box["data"] = y_pred[k]
        #    else:
        #        for k_ in y_pred[k]:
        #            if k_ in box:
        #                print("Warning: overwriting key", k_, "in semantic info")
        #            box[k_] = y_pred[k][k_]
        #    new_y[k] = box
        # y_pred = new_y

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

from collections.abc import Generator

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.refactor.base import BaseGraphPLModule


class BaseRolloutForecastingPLModule(BaseGraphPLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout = self.sample_static_info.rollout_info()

    def get_input_from_batch(self, batch, **kwargs):
        return self.rollout.next_input(step="0h", database=batch, previous_input=None, previous_output=None)

    def get_target_from_batch(self, batch, **kwargs):
        return self.rollout.next_target(step="0h", database=batch, previous_input=None, previous_output=None)

    def get_semantic_from_static_info(self, static_info, target, **kwargs):
        # get semantic information from target (should use static info)
        target_static_info = static_info["target"]
        semantic = target_static_info.new_empty()
        for k, v in target_static_info.items():
            box = v.copy()
            if "data" in v:
                v.pop("data")
            # allows to look for some information in the target
            if "latitudes" not in v and "latitudes" in target[k]:
                v["latitudes"] = target[k]["latitudes"]
            if "longitudes" not in v and "longitudes" in target[k]:
                v["longitudes"] = target[k]["longitudes"]
            if "timedeltas" not in v and "timedeltas" in target[k]:
                v["timedeltas"] = target[k]["timedeltas"]
            if "reference_date" in target[k]:
                v["reference_date"] = target[k]["reference_date"]
            if "reference_date_str" in target[k]:
                v["reference_date_str"] = target[k]["reference_date_str"]
            semantic[k] = box
        return semantic

    def _step(
        self,
        batch,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        print("️⚠️💬 Starting step on Rollout")
        static_info = self.model.sample_static_info

        # merge batch with static data
        batch = static_info + batch

        batch = self.apply_normaliser_to_batch(batch)

        loss = torch.zeros(1, dtype=batch.first["data"].dtype, device=self.device, requires_grad=True)

        # get input and target
        input = self.get_input_from_batch(batch)
        target = self.get_target_from_batch(batch)
        print(input.to_str("⚠️input data"))
        print(target.to_str("⚠️target data"))

        semantic = self.get_semantic_from_static_info(static_info, target)
        print(semantic.to_str("⚠️semantic info from target"))

        # graph = self.graph_editor.update_graph(self.graph_data, input_latlons, target_latlons)

        # run model for one step
        y_pred = self(input, self.graph_data.clone().to("cuda"))
        # y_pred = target.select_content(["data"])  # for development, don't keep this line
        print(y_pred.to_str("⚠️y_pred before merging semantic info from target"))

        # compute loss

        # y_pred = semantic + y_pred
        new_y = semantic.new_empty()
        for k, v in semantic.items():
            box = v.copy()
            for k_ in y_pred[k]:
                if k_ in box:
                    print("Warning: overwriting key", k_, "in semantic info")
                box[k_] = y_pred[k][k_]
            new_y[k] = box
        y_pred = new_y

        print(y_pred.to_str("⚠️y_pred after merging semantic info from target"))
        loss = 0
        for k, module in self.loss.items():
            loss += module(pred=y_pred[k], target=target[k])
        print("computed loss:", loss)
        assert False, "stop here"

        # Iterate over all entries in batch["target"] and accumulate loss
        for target_key, target_data in batch["target"].items():
            loss += checkpoint(
                self.loss,
                y_pred[target_key].unsqueeze(0),  # add batch dimension, why do we not get this from the model?
                target_data["data"].permute(0, 2, 1),
                use_reentrant=False,
            )  # weighting will probably not be correct here ...
        loss *= 1 / len(batch["target"])  # Average loss over all targets

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(y_pred, batch["target"], rollout_step=0)

        print(f"computed loss: {loss}, metrics: {metrics_next}, y_pred: {y_pred.to_str('y_pred')}")
        print("️⚠️💬 End of _step")
        return loss, metrics_next, y_pred


# class _RolloutForecastingPLModule(ForecastingPLModule):
#     def select_rollout_target(self, batch: dict, rollout_step: int) -> dict:
#         return batch["target"][rollout_step]
#
#     def advance_input(
#         self,
#         x,
#         y_pred,
#         batch,
#         rollout_step: int,
#     ) -> dict[str, torch.Tensor]:
#         num_target_time = 1  # If f(x_t-1, x_t) = [x_t+1, x_t+2], we should rollout 2 steps instead of 1.
#         x = x.roll(-num_target_time, dims=1)  # Roll accross TIME dim
#
#         # Get prognostic variables
#         x[-num_target_time:, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
#             ...,
#             self.data_indices.internal_model.output.prognostic,
#         ]
#
#         # get new "constants" needed for time-varying fields
#         x[-num_target_time, :, :, self.data_indices.internal_model.input.forcing] = batch["input"][
#             :,
#             self.multi_step + rollout_step,
#             :,
#             :,
#             self.data_indices.internal_data.input.forcing,
#         ]
#         return x
#
#     def _step(self, batch, validation_mode: bool = False):
#         batch = self.process_batch(batch)
#         x = {"input": batch["input"]}
#
#         loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
#         metrics = {}
#         y_preds = []
#         for rollout_step in range(self.rollout):
#             x["target"] = self.select_rollout_target(batch, rollout_step)
#
#             loss_next, metrics_next, y_preds_next = super()._step(
#                 x,
#                 validation_mode=validation_mode,
#                 apply_processors=False,
#             )
#
#             loss += loss_next
#             metrics.update(metrics_next)
#             y_preds.append(y_preds_next)
#
#             x["input"] = self.advance_input(x["input"], batch, rollout_step)
#
#         loss *= 1.0 / self.rollout
#         return loss, metrics, y_preds


class RolloutForecastingPLModule(BaseRolloutForecastingPLModule):
    pass


class ToyRolloutForecastingPLModule(BaseRolloutForecastingPLModule):
    pass

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.refactor.base import BaseGraphPLModule

if TYPE_CHECKING:
    from anemoi.training.data.refactor.structure import NestedTensor


class ForecastingPLModule(BaseGraphPLModule):
    def get_input_from_batch(self, batch, **kwargs):
        return batch["input"]

    def get_target_from_batch(self, batch, **kwargs):
        return batch["target"]

    def get_semantic_from_static_info(self, batch_staticinfo, target, **kwargs):
        # get semantic information from target (should use static info)
        target_static_info = batch_staticinfo["target"]
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
        self, batch: "NestedTensor", validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)
        """
        # ✅ here, batch = input + target
        # validation vs training, do we have validation batch or training batch?
        #
        # we should create a sample_provider in AnemoiTrainer?
        # or in this module then give it to the dataloader and dataset?
        print("️⚠️💬 Starting _step")
        static_info = self.model.sample_static_info
        batch_staticinfo = self.batch_staticinfo

        # merge batch with static data
        batch = batch_staticinfo + batch

        assert batch

        print(batch.to_str("⚠️batch before normalistation"))
        for k, v in batch.items():
            normaliser = self.normaliser[k]
            assert isinstance(normaliser, torch.nn.Module), type(normaliser)
            v["data"] = normaliser(v["data"])
        # Could be done with:
        # batch.each["data"] = self.normaliser.each(batch.each["data"])
        print(batch.to_str("⚠️batch after normalistation"))

        loss = torch.zeros(1, dtype=batch.first["data"].dtype, device=self.device, requires_grad=True)
        print(self.loss.to_str("⚠️loss function"))

        input = self.get_input_from_batch(batch)
        target = self.get_target_from_batch(batch)
        print(input.to_str("⚠️input data"))
        print(target.to_str("⚠️target data"))

        semantic = self.get_semantic_from_static_info(batch_staticinfo, target)
        print(semantic.to_str("⚠️semantic info from target"))

        # graph = self.graph_editor.update_graph(self.graph_data, input_latlons, target_latlons)

        # run model for one step
        y_pred = self(input, self.graph_data.clone().to("cuda"))
        # y_pred = target.select_content(["data"])  # for development, don't keep this line
        print(y_pred.to_str("⚠️y_pred before merging semantic info from target"))

        # compute loss
        y_pred = semantic + y_pred
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

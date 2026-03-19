import torch
import os
from anemoi.training.train.tasks.diffusionforecaster import GraphUnconditionalDiffusionForecaster
model = torch.load("/project/home/p200177/DE_371/avritj/experiments_anemoi/training/overfit_x=0_1_image_1024/checkpoint/fa49360a15c44366bc28dd0f829553fc/anemoi-by_time-epoch_023-step_025000.ckpt", weights_only = False)
print(model.keys())
ckpt = "/project/home/p200177/DE_371/avritj/experiments_anemoi/training/overfit_x=0_1_image_1024/checkpoint/fa49360a15c44366bc28dd0f829553fc/last.ckpt"

f = GraphUnconditionalDiffusionForecaster.load_from_checkpoint(
    ckpt,
    weights_only=False
)
print("ouais ouais la loss",f.loss)
# state_dict = model["state_dict"]
# scaler_keys = [k for k in state_dict.keys() if any(
#     name in k for name in ['pressure_level', 'general_variable', 'node_weights']
# )]
# for k in scaler_keys:
#     print(k, state_dict[k].shape)

# hp = model["hyper_parameters"]
# scaler_keys=[k for k in hp.keys() if any(
#     name in k for name in ['pressure_level', 'general_variable', 'node_weights']
# )]
# for k in scaler_keys:
#     print(k, hp[k])

# def forward(
#         self,
#         pred: torch.Tensor,
#         target: torch.Tensor,
#         weights: torch.Tensor | None = None,
#         squash: bool = True,
#         *,
#         scaler_indices: tuple[int, ...] | None = None,
#         without_scalers: list[str] | list[int] | None = None,
#         grid_shard_slice: slice | None = None,
#         group: ProcessGroup | None = None,
#     ) -> torch.Tensor:
#         """Calculates the weighted MSE loss.

#         Parameters
#         ----------
#         pred : torch.Tensor
#             Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
#         target : torch.Tensor
#             Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
#         weights : torch.Tensor | None, optional
#             Weights to apply to the MSE difference, by default None
#         squash : bool, optional
#             Average last dimension, by default True
#         scaler_indices: tuple[int,...], optional
#             Indices to subset the calculated scaler with, by default None
#         without_scalers: list[str] | list[int] | None, optional
#             list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
#             By default None
#         grid_shard_slice : slice, optional
#             Slice of the grid if x comes sharded, by default None
#         group: ProcessGroup, optional
#             Distributed group to reduce over, by default None

#         Returns
#         -------
#         torch.Tensor
#             Weighted MSE loss
#         """
#         is_sharded = grid_shard_slice is not None
#         out = self.calculate_difference(pred, target)
#         if weights is not None:
#             out = out * weights
#         # rank_zero_info(f"dans MSE weighted : scaler indices : {scaler_indices}; without scalers = {without_scalers} and squash : {squash}", )
#         # rank_zero_info(f"loss avant scaling :  {out}")
#         out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
#         # rank_zero_info(f"loss après scaling :  {out}")
#         out_2 = self.reduce(out, squash, group=group if is_sharded else None)
#         # rank_zero_info(f"loss aprèes reduce :  {out_2}")
#         return self.reduce(out, squash, group=group if is_sharded else None)

# # def scale(
# #         self,
#         x: torch.Tensor,
#         subset_indices: tuple[int, ...] | None = None,
#         *,
#         without_scalers: list[str] | list[int] | None = None,
#         grid_shard_slice: slice | None = None,
#     ) -> torch.Tensor:
#         """Scale a tensor by the variable_scaling.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs)
#         subset_indices: tuple[int,...], optional
#             Indices to subset the calculated scaler and `x` tensor with, by default None.
#         without_scalers: list[str] | list[int] | None, optional
#             list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
#             By default None
#         grid_shard_slice : slice, optional
#             Slice of the grid if x comes sharded, by default None

#         Returns
#         -------
#         torch.Tensor
#             Scaled error tensor
#         """

#         rank_zero_info("############ info in scale ################")
       
#         if subset_indices is None:
#             subset_indices = [Ellipsis]

#         if len(self.scaler) == 0:
#             return x[subset_indices]

#         if TensorDim.GRID not in self.scaler:
#             error_msg = (
#                 "Scaler tensor must be at least applied to the GRID dimension. "
#                 "Please add a scaler here, use `UniformWeights` for simple uniform scaling.",
#             )
#             raise RuntimeError(error_msg)

#         scale_tensor = self.scaler
#         if without_scalers is not None and len(without_scalers) > 0:
#             if isinstance(without_scalers[0], str):
#                 scale_tensor = self.scaler.without(without_scalers)
#             else:
#                 scale_tensor = self.scaler.without_by_dim(without_scalers)
#         rank_zero_info(f"subset indices : {subset_indices}")
#         rank_zero_info(f"scale tensor shape {scale_tensor.shape}")
#         rank_zero_info(f"scaler : {self.scaler}")
#         rank_zero_info(f"without scaler : {without_scalers}")
#         breakpoint()
#         return scale_tensor.scale_iteratively(
#             x,
#             subset_indices=subset_indices,
#             grid_shard_slice=grid_shard_slice,
#         )

#     def reduce(
#         self,
#         out: torch.Tensor,
#         squash: bool = True,
#         squash_mode: str = "avg",
#         group: ProcessGroup | None = None,
#     ) -> torch.Tensor:
#         """Reduce the out of the loss.

#         If `squash` is True, the last dimension is averaged.

#         Irrespective of `squash`, the output is reduced over the
#         batch, ensemble and grid dimensions.

#         Parameters
#         ----------
#         out : torch.Tensor
#             Difference tensor, of shape TensorDim
#         squash : bool, optional
#             Whether to squash the variable dimension, by default True
#         squash_mode : str, optional
#             Mode to use for squashing the variable dimension, by default "avg"
#             If "avg", the last dimension is averaged.
#             If "sum", the last dimension is summed.

#         Returns
#         -------
#         torch.Tensor
#             Reduced output tensor

#         Raises
#         ------
#         ValueError
#             If squash_mode is not one of ['avg', 'sum']
#         """
#         if squash:
#             if squash_mode == "avg":
#                 print("avg in squash")
#                 out = self.avg_function(out, dim=TensorDim.VARIABLE)
#             elif squash_mode == "sum":
#                 print("sum in squash")
#                 out = self.sum_function(out, dim=TensorDim.VARIABLE)
#             else:
#                 msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
#                 raise ValueError(msg)

#         # here the grid dimension is summed because the normalisation is handled in the node weighting
#         grid_summed = self.sum_function(out, dim=(TensorDim.GRID))
#         out = self.avg_function(
#             grid_summed,
#             dim=(
#                 TensorDim.BATCH_SIZE,
#                 TensorDim.ENSEMBLE_DIM,
#             ),
#         )

#         return out if group is None else reduce_tensor(out, group)

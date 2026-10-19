# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import MultiscaleLossWrapper


def _multiscale(per_scale_loss: torch.nn.Module) -> MultiscaleLossWrapper:
    return MultiscaleLossWrapper(per_scale_loss=per_scale_loss, weights=[1.0], multiscale_config=None)


def test_wrapper_shares_scaler_of_single_loss() -> None:
    mse = MSELoss()
    wrapper = _multiscale(mse)
    assert wrapper.scaler is mse.scaler


def test_multiscale_around_combined_matches_combined_of_multiscales() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 1, 10, 4)
    target = torch.randn(2, 3, 1, 10, 4)

    separate = CombinedLoss(_multiscale(MSELoss()), _multiscale(MAELoss()), loss_weights=(0.6, 0.4))
    shared = _multiscale(CombinedLoss(MSELoss(), MAELoss(), loss_weights=(0.6, 0.4)))

    torch.testing.assert_close(shared(pred, target), separate(pred, target))


def test_scalers_reach_sub_losses_through_wrapper() -> None:
    mse, mae = MSELoss(), MAELoss()
    wrapper = _multiscale(CombinedLoss(mse, mae))
    assert not hasattr(wrapper, "scaler")

    wrapper.add_scaler(3, torch.ones(10), name="node_weights")
    assert "node_weights" in mse.scaler
    assert "node_weights" in mae.scaler

    wrapper.update_scaler("node_weights", torch.full((10,), 2.0))
    torch.testing.assert_close(mse.scaler.get_scaler_tensor("node_weights"), torch.full((10,), 2.0))
    torch.testing.assert_close(mae.scaler.get_scaler_tensor("node_weights"), torch.full((10,), 2.0))

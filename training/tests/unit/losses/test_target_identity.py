# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import TargetIdentityLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.utils import check_loss_tree_variable_units
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

NAMES = ["lsm", "z_500", "t_500", "u_500", "era_z_500", "era_t_500"]
N2I = {n: i for i, n in enumerate(reversed(NAMES))}  # scrambled data order


def _indices() -> IndexCollection:
    return IndexCollection(
        DictConfig({"forcing": ["lsm"], "diagnostic": [], "target": ["era_z_500", "era_t_500"]}),
        N2I,
    )


def _normalizer(di: IndexCollection) -> SimpleNamespace:
    mul = torch.ones(len(N2I))
    add = torch.zeros(len(N2I))
    mul[di.name_to_index["z_500"]], add[di.name_to_index["z_500"]] = 1e-4, -5.0  # min-max like
    mul[di.name_to_index["t_500"]], add[di.name_to_index["t_500"]] = 0.1, -25.0  # mean-std like
    mul[di.name_to_index["u_500"]], add[di.name_to_index["u_500"]] = 0.08, 0.0
    return SimpleNamespace(_norm_mul=mul, _norm_add=add)


def _loss(di: IndexCollection, norm: SimpleNamespace, **kwargs) -> TargetIdentityLoss:
    pairs = [{"target": "era_z_500", "model": "z_500", "weight": 2.0}, {"target": "era_t_500", "model": "t_500"}]
    defaults = {"pairs": pairs}
    defaults.update(kwargs)
    loss = TargetIdentityLoss(data_indices=di, normalizer=norm, **defaults)
    loss.add_scaler(TensorDim.GRID, torch.full((4,), 0.25), name="node_weights")
    return loss


def _tensors(
    di: IndexCollection,
    norm: SimpleNamespace,
    z_phys: torch.Tensor,
    t_phys: torch.Tensor,
    dz: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prediction = truth + offset (normalised units); target holds the physical truth."""
    grid = z_phys.shape[0]
    full = torch.zeros(1, 1, 1, grid, len(N2I))
    for name, phys in (("z_500", z_phys), ("t_500", t_phys)):
        i = di.name_to_index[name]
        full[..., i] = phys * norm._norm_mul[i] + norm._norm_add[i]
    full[..., di.name_to_index["era_z_500"]] = z_phys
    full[..., di.name_to_index["era_t_500"]] = t_phys
    pred = full[..., di.model_output_positions_in_data_full].clone()
    pos = di.model.output.name_to_position
    pred[..., pos["z_500"]] += dz
    pred[..., pos["t_500"]] += dt
    return pred, full


def test_residual_is_in_model_normalised_units_and_weighted() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    z = torch.tensor([50000.0, 55000.0, 56000.0, 58000.0])
    t = torch.tensor([250.0, 255.0, 260.0, 265.0])
    pred, full = _tensors(di, norm, z, t, dz=0.3, dt=-0.5)
    value = loss(pred, full, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
    # sonde reduction with unit-sum node weights: mean squared normalised offset per pair, weighted 2 and 1
    torch.testing.assert_close(value, torch.tensor(2.0 * 0.3**2 + 1.0 * 0.5**2), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss.last_pair_bias, torch.tensor([0.3, -0.5]), rtol=1e-4, atol=1e-6)
    assert loss.last_pair_counts.tolist() == [4, 4]


def test_sigma_mode_uses_physical_units() -> None:
    di = _indices()
    norm = _normalizer(di)
    pairs = [
        {"target": "era_z_500", "model": "z_500", "sigma": 100.0},
        {"target": "era_t_500", "model": "t_500", "sigma": 1.0},
    ]
    loss = _loss(di, norm, pairs=pairs)
    z = torch.full((4,), 55000.0)
    t = torch.full((4,), 250.0)
    # dz = 0.01 normalised = 100 m^2/s^2 physical = 1 sigma; dt = 0.2 normalised = 2 K = 2 sigma
    pred, full = _tensors(di, norm, z, t, dz=0.01, dt=0.2)
    value = loss(pred, full, target_layout="data_full")
    torch.testing.assert_close(value, torch.tensor(1.0 + 4.0), rtol=1e-3, atol=1e-4)


def test_masking_sonde_vs_per_obs() -> None:
    di = _indices()
    norm = _normalizer(di)
    z = torch.full((4,), 55000.0)
    t = torch.full((4,), 250.0)
    pred, full = _tensors(di, norm, z, t, dz=0.2, dt=0.0)
    full[..., 1:, di.name_to_index["era_z_500"]] = float("nan")  # only node 0 observed
    sonde = _loss(di, norm)(pred, full, target_layout="data_full")
    per_obs = _loss(di, norm, reduction="per_obs")(pred, full, target_layout="data_full")
    torch.testing.assert_close(sonde, torch.tensor(2.0 * 0.2**2 * 0.25), rtol=1e-4, atol=1e-7)  # one of four nodes
    torch.testing.assert_close(per_obs, torch.tensor(2.0 * 0.2**2), rtol=1e-4, atol=1e-7)


def test_layouts_gradients_and_squash_false() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm, huber_delta=1.0)
    z = torch.full((4,), 55000.0)
    t = torch.full((4,), 250.0)
    pred, full = _tensors(di, norm, z, t, dz=0.1, dt=3.0)  # dt beyond the Huber delta
    out = full[..., di.data.output.full]
    pred.requires_grad_(True)
    v_full = loss(pred, full, target_layout="data_full")
    v_out = loss(pred, out, target_layout="data_output")
    torch.testing.assert_close(v_full, v_out)
    torch.testing.assert_close(v_full, torch.tensor(2.0 * 0.1**2 + (2 * 1.0 * 3.0 - 1.0)), rtol=1e-4, atol=1e-6)
    v_full.backward()
    pos = di.model.output.name_to_position
    assert pred.grad[..., pos["z_500"]].abs().sum() > 0 and pred.grad[..., pos["t_500"]].abs().sum() > 0
    assert pred.grad[..., pos["u_500"]].abs().sum() == 0
    vec = loss(pred.detach(), full, target_layout="data_full", squash=False)
    assert vec.shape == (pred.shape[-1],) and vec[pos["u_500"]] == 0
    torch.testing.assert_close(vec.sum(), v_full.detach())


def test_all_nan_targets_give_zero_finite_loss() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm, reduction="per_obs")
    pred, full = _tensors(di, norm, torch.full((4,), 55000.0), torch.full((4,), 250.0), dz=0.1, dt=0.1)
    full[..., di.name_to_index["era_z_500"]] = float("nan")
    full[..., di.name_to_index["era_t_500"]] = float("nan")
    pred.requires_grad_(True)
    value = loss(pred, full, target_layout="data_full")
    assert value.item() == 0.0
    value.backward()
    assert torch.isfinite(pred.grad).all()


def test_construction_errors_and_factory() -> None:
    di = _indices()
    norm = _normalizer(di)
    with pytest.raises(ValueError, match="model output"):
        _loss(di, norm, pairs=[{"target": "era_z_500", "model": "z_999"}])
    with pytest.raises(ValueError, match=r"data\.output"):
        _loss(di, norm, pairs=[{"target": "nope", "model": "z_500"}])
    bad = _normalizer(di)
    bad._norm_mul[di.name_to_index["era_z_500"]] = 0.5
    with pytest.raises(ValueError, match="physical units"):
        _loss(di, bad)
    with pytest.raises(ValueError, match="sigma"):
        _loss(
            di,
            norm,
            pairs=[{"target": "era_z_500", "model": "z_500", "sigma": 1.0}, {"target": "era_t_500", "model": "t_500"}],
        )
    cfg = DictConfig(
        {
            "_target_": "anemoi.training.losses.CombinedLoss",
            "losses": [
                {"_target_": "anemoi.training.losses.MSELoss", "scalers": [], "ignore_nans": True},
                {
                    "_target_": "anemoi.training.losses.TargetIdentityLoss",
                    "scalers": [],
                    "pairs": [{"target": "era_z_500", "model": "z_500"}],
                },
            ],
            "loss_weights": [1.0, 0.5],
        },
    )
    loss = get_loss_function(cfg, scalers={}, data_indices=di, normalizer=norm)
    assert isinstance(loss, CombinedLoss) and isinstance(loss.losses[1], TargetIdentityLoss)
    assert not hasattr(loss.losses[1], "predicted_variables")
    check_loss_tree_variable_units(loss, None)

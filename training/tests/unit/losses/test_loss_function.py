# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import einops
import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import FourierCorrelationLoss
from anemoi.training.losses import HuberLoss
from anemoi.training.losses import KernelCRPS
from anemoi.training.losses import LogCoshLoss
from anemoi.training.losses import LogSpectralDistance
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import RMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses import SpectralL2Loss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim

losses = [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss]
spectral_losses = [SpectralL2Loss, SpectralCRPSLoss, FourierCorrelationLoss, LogSpectralDistance]
losses += spectral_losses


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_manual_init(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls(x_dim=4, y_dim=4) if loss_cls in spectral_losses else loss_cls()
    assert isinstance(loss, BaseLoss)


@pytest.fixture
def functionalloss() -> type[FunctionalLoss]:
    class ReturnDifference(FunctionalLoss):
        def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return pred - target

    return ReturnDifference


@pytest.fixture
def loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 1, 4, 2]

    pred = torch.zeros(tensor_shape)
    pred[0, 0, 0, 0] = torch.tensor([1.0, 1.0])
    target = torch.zeros(tensor_shape)

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


@pytest.fixture
def loss_inputs_fine(
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs with finer grid."""
    pred, target, loss_result = loss_inputs

    pred = torch.cat([pred, pred], dim=2)
    target = torch.cat([target, target], dim=2)

    return pred, target, loss_result


def test_assert_of_grid_dim(functionalloss: type[FunctionalLoss]) -> None:
    """Test that the grid dimension is set correctly."""
    loss = functionalloss()
    loss.add_scaler(TensorDim.VARIABLE, 1.0, name="variable_test")

    assert TensorDim.GRID not in loss.scaler, "Grid dimension should not be set"

    with pytest.raises(RuntimeError):
        loss.scale(torch.ones((4, 2)))


@pytest.mark.parametrize("add_grid_scaler", [False, True])
def test_scale_subset_indices_requires_tuple(
    functionalloss: type[FunctionalLoss],
    add_grid_scaler: bool,
) -> None:
    loss = functionalloss()
    if add_grid_scaler:
        loss.add_scaler(TensorDim.GRID, torch.tensor([1.0, 2.0, 3.0, 4.0]), name="grid_test")

    x = torch.arange(1 * 1 * 1 * 4 * 5, dtype=torch.float32).reshape(1, 1, 1, 4, 5)
    with pytest.raises(TypeError, match="must be a tuple"):
        loss.scale(x, subset_indices=[Ellipsis, [1, 3]])


@pytest.fixture
def simple_functionalloss(functionalloss: type[FunctionalLoss]) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.ones((4,)), name="unit_scaler")
    return loss


@pytest.fixture
def functionalloss_with_scaler(simple_functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = simple_functionalloss
    loss.add_scaler(TensorDim.GRID, torch.rand((4,)), name="test")
    return loss


@pytest.fixture
def functionalloss_with_scaler_fine(functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.rand((8,)), name="test")
    return loss


def test_simple_functionalloss(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test a functional loss."""
    pred, target, loss_result = loss_inputs

    loss = simple_functionalloss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_result), "Loss should be equal to the expected result"


def test_batch_invariance(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, loss_result = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 4

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert torch.allclose(loss_batch_size_1, loss_result), "Loss should be equal to the expected result"

    assert isinstance(loss_batch_size_2, torch.Tensor)
    assert torch.allclose(loss_batch_size_2, loss_result), "Loss should be equal to the expected result"

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_without_squash(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1, squash=False)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2, squash=False)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_with_scaler(
    functionalloss_with_scaler: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape

    loss_batch_size_1 = functionalloss_with_scaler(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = functionalloss_with_scaler(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_grid_invariance(
    functionalloss_with_scaler: FunctionalLoss,
    functionalloss_with_scaler_fine: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    gdim = TensorDim.GRID
    pred_coarse, target_coarse, _ = loss_inputs
    pred_fine = torch.cat([pred_coarse, pred_coarse], dim=gdim)
    target_fine = torch.cat([target_coarse, target_coarse], dim=gdim)

    num_points_coarse = pred_coarse.shape[gdim]
    num_points_fine = pred_fine.shape[gdim]

    functionalloss_with_scaler.update_scaler("test", torch.ones((num_points_coarse,)) / num_points_coarse)
    functionalloss_with_scaler_fine.update_scaler("test", torch.ones((num_points_fine,)) / num_points_fine)

    loss_coarse = functionalloss_with_scaler(pred_coarse, target_coarse)
    loss_fine = functionalloss_with_scaler_fine(pred_fine, target_fine)

    assert isinstance(loss_coarse, torch.Tensor)
    assert isinstance(loss_fine, torch.Tensor)

    assert torch.allclose(loss_coarse, loss_fine), "Losses should be equal between grid sizes"


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_include(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "x_dim": 4,
            "y_dim": 4,
        }
    )
    loss = get_loss_function(DictConfig(loss_dic))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["test"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["test"],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_add_all(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*"],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler_not_add(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": [],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": [],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler_exclude(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*", "!test"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "x_dim": 4,
            "y_dim": 4,
            "scalers": ["*", "!test"],
        }
    )
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


def test_logfft2dist_loss() -> None:
    """Test that LogFFT2Distance can be instantiated and validates input shape."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.LogFFT2Distance",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, BaseLoss)
    assert hasattr(loss.transform, "x_dim")
    assert hasattr(loss.transform, "y_dim")

    # pred/target are (batch, steps, grid, vars)
    right = (torch.ones((6, 1, 1, 710 * 640, 2)), torch.zeros((6, 1, 1, 710 * 640, 2)))

    # squash=False -> per-variable loss
    loss_value = loss(*right, squash=False)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 1 and loss_value.shape[0] == 2, "Expected per-variable loss (n_vars,)"

    # squash=True -> single aggregated loss
    loss_total = loss(*right, squash=True)
    assert isinstance(loss_total, torch.Tensor)
    assert loss_total.numel() == 1, "Expected a single aggregated loss value"

    # wrong grid size should fail (FFT2D reshape/assert)
    wrong = (torch.ones((6, 1, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 1, 710 * 640 + 1, 2)))
    with pytest.raises(einops.EinopsError):
        _ = loss(*wrong, squash=True)


def test_fcl_loss() -> None:
    """Test that FourierCorrelationLoss can be instantiated and validates input shape."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.FourierCorrelationLoss",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, BaseLoss)
    assert hasattr(loss.transform, "x_dim")
    assert hasattr(loss.transform, "y_dim")

    right = (torch.ones((6, 1, 1, 710 * 640, 2)), torch.zeros((6, 1, 1, 710 * 640, 2)))

    loss_value = loss(*right, squash=False)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 1 and loss_value.shape[0] == 2, "Expected per-variable loss (n_vars,)"

    loss_total = loss(*right, squash=True)
    assert isinstance(loss_total, torch.Tensor)
    assert loss_total.numel() == 1, "Expected a single aggregated loss value"

    wrong = (torch.ones((6, 1, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 1, 710 * 640 + 1, 2)))
    with pytest.raises(einops.EinopsError):
        _ = loss(*wrong, squash=True)


def test_iter_leaf_losses_flat() -> None:
    """Test that iter_leaf_losses on a simple loss yields itself."""
    loss = MSELoss()
    leaves = list(loss.iter_leaf_losses())
    assert len(leaves) == 1
    assert leaves[0] is loss


def test_octahedral_sht_loss() -> None:
    def _octahedral_expected_points(nlat: int) -> int:
        half = [4 * (i + 1) + 16 for i in range(nlat // 2)]
        nlon = half + half[::-1]
        return int(sum(nlon))

    nlat = 8
    nvars = 3
    expected_points = _octahedral_expected_points(nlat)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralL2Loss",
                "transform": "octahedral_sht",
                "nlat": nlat,
                "scalers": [],
            },
        ),
    )
    pred = torch.zeros((2, 1, 1, expected_points, nvars))
    target = torch.zeros_like(pred)
    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), "squash=False should return per-variable loss"
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, "squash=True should return a single aggregated loss"
    pred_wrong = torch.zeros((2, 1, 1, expected_points + 1, nvars))
    target_wrong = torch.zeros_like(pred_wrong)
    with pytest.raises(AssertionError):
        _ = loss(pred_wrong, target_wrong, squash=True)


def _expected_octahedral_points(truncation: int) -> int:
    # full globe reduced-octahedral points for ecTrans definition
    # NH lons: 20 + 4*i, i=0..T  => sum_NH = 2*(T+1)*(T+10)
    # full globe doubles:        => 4*(T+1)*(T+10)
    return 4 * (truncation + 1) * (truncation + 10)


def test_spectral_crps_fft_and_dct() -> None:
    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    for transform in ["fft2d", "dct2d"]:
        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                    "transform": transform,
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "scalers": [],
                },
            ),
        )

        out = loss(pred, target, squash=False)
        assert out.shape == (nvars,), f"{transform}: per-variable CRPS expected"
        out_total = loss(pred, target, squash=True)
        assert out_total.numel() == 1, f"{transform}: scalar CRPS expected"


@pytest.mark.parametrize("transform", ["fft2d", "dct2d"])
def test_spectral_crps_projection_from_file(transform: str, mocker: MockerFixture) -> None:
    from scipy.sparse import eye

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    # Create a mock sparse matrix (e.g., identity)
    sparse_mat = eye(grid, format="csr")

    # Mock scipy.sparse.load_npz to return our sparse matrix
    mocker.patch("scipy.sparse.load_npz", return_value=sparse_mat)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": transform,
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {"matrix_path": "/path/to/projection_matrix.npz"},
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), f"{transform}: per-variable CRPS expected"
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, f"{transform}: scalar CRPS expected"


def test_spectral_crps_projection_applies_nodes_slice_before_projection(mocker: MockerFixture) -> None:
    from scipy.sparse import eye

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    projected_grid = x_dim * y_dim
    source_grid = projected_grid * 2

    pred = torch.randn(bs, 1, ens, source_grid, nvars)
    target = torch.randn(bs, 1, 1, source_grid, nvars)

    mocker.patch("scipy.sparse.load_npz", return_value=eye(projected_grid, format="csr"))

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "nodes_slice": (0, projected_grid),
                "projection_config": {"matrix_path": "/path/to/projection_matrix.npz"},
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)


def test_spectral_crps_projection_from_graph_config() -> None:
    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 2, 2
    grid = x_dim * y_dim

    graph = HeteroData()
    graph["data"].x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.017453292],
            [0.017453292, 0.0],
            [0.017453292, 0.017453292],
        ],
        dtype=torch.float32,
    )
    graph["data"].num_nodes = grid

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": [0.0, 0.0, 1.0, 1.0],
                        "longitudes": [0.0, 1.0, 0.0, 1.0],
                    },
                    "num_nearest_neighbours": 1,
                    "sigma": 1.0,
                    "row_normalize": False,
                },
                "scalers": [],
            },
        ),
        graph_data=graph,
        data_node_name="data",
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)


def test_spectral_crps_projection_from_existing_edges() -> None:
    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 2, 2
    grid = x_dim * y_dim
    edges_name = ("data", "to", "projection")

    graph = HeteroData()
    graph["data"].num_nodes = grid
    graph["projection"].num_nodes = grid
    graph[edges_name].edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    graph[edges_name].gauss_weight = torch.ones(grid)

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {
                    "edges_name": edges_name,
                    "edge_weight_attribute": "gauss_weight",
                },
                "scalers": [],
            },
        ),
        graph_data=graph,
        data_node_name="data",
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)


def test_spectral_crps_projection_uses_sharded_layout(mocker: MockerFixture) -> None:
    from scipy.sparse import eye

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)
    mocker.patch("scipy.sparse.load_npz", return_value=eye(grid, format="csr"))
    mocker.patch("anemoi.training.losses.spectral.get_shard_sizes", side_effect=lambda x, dim, _group: [x.shape[dim]])
    all_to_all = mocker.patch(
        "anemoi.training.losses.spectral.all_to_all_transpose",
        side_effect=lambda x, *_args: x,
    )
    mocker.patch("anemoi.training.losses.base.reduce_tensor", side_effect=lambda x, _group: x)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {"matrix_path": "/path/to/projection_matrix.npz"},
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False, grid_shard_sizes=[grid], group=object())

    assert out.shape == (nvars,)
    assert all_to_all.call_count == 4


def test_spectral_crps_with_target_without_ensemble_dim() -> None:
    """CRPS should handle target tensors shaped [B,T,G,V] (no ensemble dim)."""
    bs, ens, nvars = 2, 4, 2
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, grid, nvars)
    target[..., 0, 0] = torch.nan

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "ignore_nans": True,
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), "squash=False should return per-variable CRPS"
    assert torch.isfinite(out).all(), "Expected finite loss with ignore_nans=True"

    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, "squash=True should return scalar CRPS"
    assert torch.isfinite(out_total).all(), "Expected finite scalar loss with ignore_nans=True"


def test_spectral_crps_octahedral_irregular_grid_ignore_nans() -> None:
    def _octahedral_expected_points(nlat: int) -> int:
        half = [20 + 4 * i for i in range(nlat // 2)]
        return int(sum(half + half[::-1]))

    bs, ens, nvars = 2, 4, 2
    nlat = 8
    points = _octahedral_expected_points(nlat)

    pred = torch.randn(bs, 1, ens, points, nvars)
    target = torch.randn(bs, 1, 1, points, nvars)
    target[..., 0, 0] = torch.nan

    loss_no_ignore = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "octahedral_sht",
                "nlat": nlat,
                "ignore_nans": False,
                "scalers": [],
            },
        ),
    )
    out_no_ignore = loss_no_ignore(pred, target, squash=True)
    assert torch.isnan(out_no_ignore).any(), "Expected NaN when ignore_nans=False and target contains NaNs"

    loss_ignore = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "octahedral_sht",
                "nlat": nlat,
                "ignore_nans": True,
                "scalers": [],
            },
        ),
    )
    out = loss_ignore(pred, target, squash=False)
    assert out.shape == (nvars,), "octahedral_sht: per-variable CRPS expected"
    assert torch.isfinite(out).all(), "Expected finite loss when ignore_nans=True"

    out_total = loss_ignore(pred, target, squash=True)
    assert out_total.numel() == 1, "octahedral_sht: scalar CRPS expected"
    assert torch.isfinite(out_total).all(), "Expected finite scalar loss when ignore_nans=True"

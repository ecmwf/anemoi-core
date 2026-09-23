# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import CRPS
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.losses.spectral_scales import build_spectral_scales
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace


class TrackingLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: object | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del pred, target, squash
        self.calls.append(
            {
                "scaler_indices": scaler_indices,
                "without_scalers": without_scalers,
                "grid_shard_slice": grid_shard_slice,
                "group": group,
                "kwargs": kwargs,
            },
        )
        return torch.tensor(1.0)


class FixedLoss(BaseLoss):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        del target, kwargs
        return pred.new_tensor(2.0) if squash else pred.new_tensor([2.0, 3.0])


class FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


@pytest.fixture
def loss_inputs_multiscale() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 2, 4, 2]  # (batch, output_steps, ens, latlon, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, 0, :, 0] = torch.tensor([1.0, 0.0])
    target = torch.zeros([tensor_shape[0], tensor_shape[1], 1, tensor_shape[3], tensor_shape[4]])

    # Only one of the two variables differs by 1 at one grid point.
    # The mean CRPS is 0.5.

    loss_result = torch.tensor(0.5)
    return pred, target, loss_result


def test_multi_scale_instantiation(
    loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test multiscale loss instantiation with single scale."""
    per_scale_loss = CRPS()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )

    pred, target, loss_result = loss_inputs_multiscale
    loss = multiscale_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_result), "Loss should be equal to the expected result"


def test_multiscale_weights_length_mismatch_raises() -> None:
    per_scale_loss = MSELoss()
    with pytest.raises(AssertionError):
        MultiscaleLossWrapper(
            per_scale_loss=per_scale_loss,
            weights=[1.0],  # 1 weight but multiscale_config gives 2 scales
            multiscale_config={"loss_matrices": [None, None]},
        )


def test_multiscale_sums_weighted_scale_losses() -> None:
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=FixedLoss(),
        weights=[0.5, 2.0],
        multiscale_config={"loss_matrices": [None, None]},
    )
    pred = torch.zeros((1, 1, 1, 2, 2))
    target = torch.zeros_like(pred)

    scalar_loss = multiscale_loss(pred, target)
    per_variable_loss = multiscale_loss(pred, target, squash=False)

    torch.testing.assert_close(scalar_loss, torch.tensor(5.0))
    torch.testing.assert_close(per_variable_loss, torch.tensor([5.0, 7.5]))


@pytest.mark.parametrize("per_scale_loss", [CRPS(), MSELoss()])
@pytest.mark.parametrize("weights", [torch.tensor([0.3, 0.7]), torch.tensor([1.0, 2.0])])
def test_multi_scale(
    loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    per_scale_loss: BaseLoss,
    weights: torch.Tensor,
    mocker: MockerFixture,
) -> None:
    """Test multiscale loss with different per-scale losses and weights."""
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 4
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 0]])
    graph[("src", "to", "dst")].edge_weight = torch.ones(8) / 2

    smoothing_provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )

    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothers",
        return_value=[None, smoothing_provider],
    )

    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=weights,
    )

    assert smoothing_provider.projection_matrix.layout == torch.sparse_csr

    pred, target, _ = loss_inputs_multiscale
    loss = multiscale_loss(pred, target, squash=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (), "squash=True should return one aggregated loss"
    loss = multiscale_loss(pred, target, squash=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (pred.shape[-1],), "squash=False should return one loss per variable"


def test_multiscale_loss_equivalent_to_per_scale_loss() -> None:
    """Test equivalence when only one scale is used."""
    tensor_shape = [1, 1, 2, 4, 1]  # (batch, output_steps, ens, latlon, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, 0, :, 0] = torch.tensor([1.0])
    target = torch.zeros([tensor_shape[0], tensor_shape[1], 1, tensor_shape[3], tensor_shape[4]])

    per_scale_loss = CRPS()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )

    loss = multiscale_loss(pred, target)
    loss_crps = per_scale_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_crps), "Loss for single/original scale should be equal to the CRPS"


def test_multiscale_forwards_layout_kwargs_to_filtered_per_scale_loss() -> None:
    """Nested per-scale filtered losses must receive layout kwargs."""
    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1})
    multiscale_loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
                "weights": [1.0],
                "loss_matrices": [None],
                "per_scale_loss": {
                    "_target_": "anemoi.training.losses.MSELoss",
                    "scalers": [],
                },
            },
        ),
        scalers={},
        data_indices=data_indices,
    )

    pred = torch.ones((1, 1, 1, 4, 2))
    target = torch.zeros((1, 1, 1, 4, 2))
    loss = multiscale_loss(
        pred,
        target,
        group=None,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


def test_multiscale_loss_preserves_single_variable_dimension() -> None:
    pred = torch.ones((1, 1, 1, 4, 1))
    target = torch.zeros_like(pred)
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=MSELoss(),
        weights=[0.25, 0.75],
        multiscale_config={"loss_matrices": [None, None]},
    )

    loss = multiscale_loss(pred, target, squash=False)

    assert loss.shape == (1,)


def test_multiscale_loss_forwards_scaler_indices() -> None:
    pred = torch.zeros((1, 1, 1, 2, 2))
    pred[0, 0, 0, 0, 0] = 10.0
    pred[0, 0, 0, 0, 1] = 1.0
    target = torch.zeros((1, 1, 1, 2, 2))

    per_scale_loss = MSELoss()
    per_scale_loss.add_scaler(TensorDim.GRID, torch.ones(2), name="grid_weights")
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )

    scaler_indices = (..., [1])
    loss = multiscale_loss(pred, target, scaler_indices=scaler_indices)
    expected = per_scale_loss(pred, target, scaler_indices=scaler_indices)

    assert torch.allclose(loss, expected)


def test_multiscale_loss_forwards_group_and_without_scalers() -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )

    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 1, 2, 1))
    sentinel_group = FakeGroup(size=1)

    multiscale_loss(
        pred,
        target,
        scaler_indices=(..., [0]),
        without_scalers=["node_weights"],
        group=sentinel_group,
    )

    assert per_scale_loss.calls == [
        {
            "scaler_indices": (..., [0]),
            "without_scalers": ["node_weights"],
            "grid_shard_slice": None,
            "group": sentinel_group,
            "kwargs": {},
        },
    ]


def test_multiscale_loss_uses_grid_shard_sizes_for_sharding(
    mocker: MockerFixture,
) -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )
    group = FakeGroup(size=2)
    grid_shard_sizes = [1, 1]
    channel_shard_sizes_pred = [1, 1]
    channel_shard_sizes_y = [1, 1]
    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 1, 2, 1))

    prepare = mocker.patch.object(
        multiscale_loss,
        "_prepare_for_smoothing",
        return_value=(pred, target, channel_shard_sizes_pred, channel_shard_sizes_y),
    )
    a2a = mocker.patch(
        "anemoi.training.losses.multiscale.all_to_all_transpose",
        side_effect=lambda x, *_args: x,
    )

    multiscale_loss(
        pred,
        target,
        group=group,
        grid_shard_sizes=grid_shard_sizes,
    )

    prepare.assert_called_once_with(pred, target, group, grid_shard_sizes)
    # Two all_to_all_transpose calls: one for y_pred_ens_tmp, one for y_tmp
    assert a2a.call_count == 2


def test_multiscale_loss_forwards_extra_kwargs() -> None:
    per_scale_loss = TrackingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=[1.0],
    )

    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 1, 2, 1))
    sentinel = object()

    multiscale_loss(
        pred,
        target,
        custom_kwarg=sentinel,
    )

    assert per_scale_loss.calls == [
        {
            "scaler_indices": None,
            "without_scalers": None,
            "grid_shard_slice": None,
            "group": None,
            "kwargs": {"custom_kwarg": sentinel},
        },
    ]


def test_deepcopy_multiscale_loss_does_not_raise(mocker: MockerFixture) -> None:
    """deepcopy(MultiscaleLossWrapper) must not raise with CSR smoothing matrices."""
    import copy

    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 4
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph[("src", "to", "dst")].edge_weight = torch.ones(4)

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothers",
        return_value=[provider],
    )
    loss = MultiscaleLossWrapper(per_scale_loss=MSELoss(), weights=[1.0])
    copy.deepcopy(loss)  # must not raise NotImplementedError on CSR tensors


class CapturingLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash: bool = True, **kwargs: object) -> torch.Tensor:
        del squash, kwargs
        self.preds.append(pred)
        self.targets.append(target)
        return pred.new_tensor(0.0)


# An octahedral grid small enough that every ring resolves the zonal wavenumbers
# used below, so the transforms are exact up to rounding.
_SPECTRAL_NLAT = 16
_SPECTRAL_CONFIG = {"transform": "octahedral_sht", "nlat": _SPECTRAL_NLAT, "cutoffs": [3, 7]}
_PLANAR_X_DIM, _PLANAR_Y_DIM = 16, 8


def test_spherical_harmonic_scales_are_nested_projections() -> None:
    scales = build_spectral_scales("octahedral_sht", [3, 7], nlat=_SPECTRAL_NLAT)
    coarse_only = build_spectral_scales("octahedral_sht", [3], nlat=_SPECTRAL_NLAT)
    x = torch.randn(2, 1, 3, sum(octahedral_lons_per_lat(_SPECTRAL_NLAT)), 4)

    coarse = scales.synthesise(scales.analyse(x), 3)
    fine = scales.synthesise(scales.analyse(x), 7)

    assert coarse.shape == x.shape
    assert not torch.allclose(coarse, x)
    # The coarse scale from the shared analysis matches an analysis at its own truncation.
    torch.testing.assert_close(coarse, coarse_only.synthesise(coarse_only.analyse(x), 3))
    torch.testing.assert_close(scales.synthesise(scales.analyse(coarse), 3), coarse)
    torch.testing.assert_close(scales.synthesise(scales.analyse(fine), 3), coarse)


def test_spherical_harmonic_scales_need_integer_cutoffs() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_spectral_scales("octahedral_sht", [3.5], nlat=_SPECTRAL_NLAT)


@pytest.mark.parametrize("transform", ["fft2d", "dct2d"])
def test_planar_scales_keep_only_low_frequencies(transform: str) -> None:
    y = torch.arange(_PLANAR_Y_DIM, dtype=torch.float32)[:, None]
    x = torch.arange(_PLANAR_X_DIM, dtype=torch.float32)[None, :]
    if transform == "fft2d":
        # One wave across x (frequency 1/16) and four across y (frequency 1/2).
        low = torch.cos(2 * torch.pi * x / _PLANAR_X_DIM).expand(_PLANAR_Y_DIM, -1)
        high = torch.cos(2 * torch.pi * 4 * y / _PLANAR_Y_DIM).expand(-1, _PLANAR_X_DIM)
    else:
        # Cosine 1 along x (frequency 1/32) and cosine 6 along y (frequency 3/8).
        low = torch.cos(torch.pi * (x + 0.5) / _PLANAR_X_DIM).expand(_PLANAR_Y_DIM, -1)
        high = torch.cos(torch.pi * 6 * (y + 0.5) / _PLANAR_Y_DIM).expand(-1, _PLANAR_X_DIM)
    as_field = lambda f: f.reshape(1, 1, 1, -1, 1)  # noqa: E731
    scales = build_spectral_scales(transform, [0.1, 0.25], x_dim=_PLANAR_X_DIM, y_dim=_PLANAR_Y_DIM)

    coarse = scales.synthesise(scales.analyse(as_field(low + high)), 0.1)

    torch.testing.assert_close(coarse, as_field(low), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_SPECTRAL_CONFIG, id="octahedral_sht"),
        pytest.param(
            {"transform": "fft2d", "x_dim": _PLANAR_X_DIM, "y_dim": _PLANAR_Y_DIM, "cutoffs": [0.1, 0.25]},
            id="fft2d",
        ),
        pytest.param(
            {"transform": "dct2d", "x_dim": _PLANAR_X_DIM, "y_dim": _PLANAR_Y_DIM, "cutoffs": [0.1, 0.25]},
            id="dct2d",
        ),
    ],
)
def test_multiscale_spectral_bands_sum_to_input(config: dict) -> None:
    capturing_loss = CapturingLoss()
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=capturing_loss,
        weights=[1.0, 1.0, 1.0],
        multiscale_config=config,
    )
    assert multiscale_loss.smoothers == [*config["cutoffs"], None]
    # The transforms are submodules, so they follow the loss onto the GPU.
    assert any(module is multiscale_loss.spectral_scales for module in multiscale_loss.modules())

    if config["transform"] == "octahedral_sht":
        grid_size = sum(octahedral_lons_per_lat(_SPECTRAL_NLAT))
    else:
        grid_size = _PLANAR_X_DIM * _PLANAR_Y_DIM
    pred = torch.randn(1, 1, 2, grid_size, 3)
    target = torch.randn(1, 1, 1, grid_size, 3)
    multiscale_loss(pred, target)

    scales = multiscale_loss.spectral_scales
    coarsest = config["cutoffs"][0]
    torch.testing.assert_close(capturing_loss.preds[0], scales.synthesise(scales.analyse(pred), coarsest))
    torch.testing.assert_close(torch.stack(capturing_loss.preds).sum(dim=0), pred)
    torch.testing.assert_close(torch.stack(capturing_loss.targets).sum(dim=0), target)


def test_multiscale_spectral_cutoffs_must_increase() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MultiscaleLossWrapper(
            per_scale_loss=MSELoss(),
            weights=[1.0, 1.0, 1.0],
            multiscale_config={**_SPECTRAL_CONFIG, "cutoffs": [7, 3]},
        )

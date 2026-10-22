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

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources.gridded import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses import CRPS
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace


def _view(data: torch.Tensor) -> GriddedSource:
    """Attach metadata for the gridded loss fixtures."""
    return GriddedSource(
        name="data",
        data=data,
        variables=[f"v{i}" for i in range(data.shape[-1])],
        statistics={},
        coordinates=torch.zeros(data.shape[-2], 2, device=data.device),
        layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
    )


class TrackingLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        pred: GriddedSource,
        target: GriddedSource,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: object | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(pred, GriddedSource)
        assert isinstance(target, GriddedSource)
        del squash
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
        pred: GriddedSource,
        target: GriddedSource,
        squash: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        del target, kwargs
        return pred.data.new_tensor(2.0) if squash else pred.data.new_tensor([2.0, 3.0])


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
    loss = multiscale_loss(_view(pred), _view(target))

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

    scalar_loss = multiscale_loss(_view(pred), _view(target))
    per_variable_loss = multiscale_loss(_view(pred), _view(target), squash=False)

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
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[None, smoothing_provider],
    )

    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=weights,
    )

    assert smoothing_provider.projection_matrix.layout == torch.sparse_csr

    pred, target, _ = loss_inputs_multiscale
    loss = multiscale_loss(_view(pred), _view(target), squash=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (), "squash=True should return one aggregated loss"
    # Original impulse: loss 0.5. Smoothing residual: two values of magnitude 0.5.
    residual_loss = 0.5 if isinstance(per_scale_loss, CRPS) else 0.25
    torch.testing.assert_close(loss, 0.5 * weights[0] + residual_loss * weights[1])
    loss = multiscale_loss(_view(pred), _view(target), squash=False)

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

    loss = multiscale_loss(_view(pred), _view(target))
    loss_crps = per_scale_loss(_view(pred), _view(target))

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
        _view(pred),
        _view(target),
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

    loss = multiscale_loss(_view(pred), _view(target), squash=False)

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
    loss = multiscale_loss(_view(pred), _view(target), scaler_indices=scaler_indices)
    expected = per_scale_loss(_view(pred), _view(target), scaler_indices=scaler_indices)

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
        _view(pred),
        _view(target),
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
        _view(pred),
        _view(target),
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
        _view(pred),
        _view(target),
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


class RecordingNaNLoss(BaseLoss):
    """Per-scale loss that ignores NaNs and records the targets it receives."""

    def __init__(self) -> None:
        super().__init__(ignore_nans=True)
        self.targets: list[torch.Tensor] = []

    def forward(
        self,
        pred: GriddedSource,
        target: GriddedSource,
        squash: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        del pred, squash, kwargs
        self.targets.append(target.data.clone())
        return torch.tensor(0.0)


def _ring_smoother() -> ProjectionGraphProvider:
    """Smoother over a 4-node ring: each node averages itself and one neighbour."""
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 4
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 0]])
    graph[("src", "to", "dst")].edge_weight = torch.ones(8) / 2
    return ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )


def _nan_target_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random ensemble prediction and a target with one missing value at grid point 1."""
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn((1, 1, 2, 4, 2), generator=generator)
    target = torch.randn((1, 1, 1, 4, 2), generator=generator)
    nan_mask = torch.zeros_like(target, dtype=torch.bool)
    nan_mask[0, 0, 0, 1, 0] = True
    return pred, target.masked_fill(nan_mask, float("nan")), nan_mask


def test_multiscale_masks_only_original_nan_targets(mocker: MockerFixture) -> None:
    """NaN targets and the matching predictions are filled before smoothing and masked per scale, without a halo."""
    provider = _ring_smoother()
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[provider, None],
    )
    weights = [1.0, 2.0]
    multiscale_loss = MultiscaleLossWrapper(per_scale_loss=CRPS(ignore_nans=True), weights=weights)
    pred, target, nan_mask = _nan_target_inputs()

    loss = multiscale_loss(_view(pred), _view(target))

    # Expected: smooth the 0-filled prediction and target densely, then mask only the original NaN point.
    smoothing = provider.projection_matrix.to_dense()
    filled_pred = pred.masked_fill(nan_mask, 0.0)
    filled = target.masked_fill(nan_mask, 0.0)
    smooth_pred = torch.einsum("gh,btehc->btegc", smoothing, filled_pred)
    smooth_target = torch.einsum("gh,btehc->btegc", smoothing, filled)
    bands = [(smooth_pred, smooth_target), (filled_pred - smooth_pred, filled - smooth_target)]
    crps = CRPS(ignore_nans=True)
    expected = sum(
        weight * crps(_view(band_pred), _view(band_target.masked_fill(nan_mask, float("nan"))))
        for weight, (band_pred, band_target) in zip(weights, bands, strict=True)
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)


def test_multiscale_prediction_at_nan_target_does_not_affect_loss(mocker: MockerFixture) -> None:
    """The prediction where the target is missing reaches neither the loss nor its neighbours."""
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[_ring_smoother(), None],
    )
    multiscale_loss = MultiscaleLossWrapper(per_scale_loss=CRPS(ignore_nans=True), weights=[1.0, 2.0])
    pred, target, nan_mask = _nan_target_inputs()
    pred.requires_grad_(True)

    loss = multiscale_loss(_view(pred), _view(target))
    loss.backward()
    shifted = multiscale_loss(_view(pred.detach().masked_fill(nan_mask, 100.0)), _view(target))

    pred_nan_mask = nan_mask.expand_as(pred)
    assert torch.all(pred.grad[pred_nan_mask] == 0)
    assert torch.any(pred.grad[~pred_nan_mask] != 0)
    torch.testing.assert_close(shifted, loss.detach())


def test_multiscale_smooths_in_input_dtype(mocker: MockerFixture) -> None:
    """A float32 smoothing matrix is applied to float64 inputs in float64."""
    provider = _ring_smoother()
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[provider, None],
    )
    multiscale_loss = MultiscaleLossWrapper(per_scale_loss=CRPS(), weights=[1.0, 2.0])
    pred, target, _ = _nan_target_inputs()
    pred, target = pred.double(), torch.nan_to_num(target).double()

    loss = multiscale_loss(_view(pred), _view(target))

    assert loss.dtype == torch.float64
    assert provider.projection_matrix.dtype == torch.float64
    assert torch.isfinite(loss)


def test_multiscale_nan_fill_leaves_finite_targets_unchanged(mocker: MockerFixture) -> None:
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[_ring_smoother(), None],
    )
    pred, target, nan_mask = _nan_target_inputs()
    target = target.masked_fill(nan_mask, 0.5)

    loss_ignoring_nans = MultiscaleLossWrapper(per_scale_loss=CRPS(ignore_nans=True), weights=[1.0, 2.0])
    loss_plain = MultiscaleLossWrapper(per_scale_loss=CRPS(), weights=[1.0, 2.0])

    assert torch.equal(
        loss_ignoring_nans(_view(pred), _view(target)),
        loss_plain(_view(pred), _view(target)),
    )


def test_multiscale_nan_target_without_ignore_nans_gives_nan(mocker: MockerFixture) -> None:
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[_ring_smoother(), None],
    )
    multiscale_loss = MultiscaleLossWrapper(per_scale_loss=CRPS(), weights=[1.0, 2.0])
    pred, target, _ = _nan_target_inputs()

    assert torch.isnan(multiscale_loss(_view(pred), _view(target)))


def test_multiscale_sharded_smooths_filled_target_and_restores_nans(mocker: MockerFixture) -> None:
    """With sharding, the filled prediction and target are smoothed and NaNs are restored in the grid-sharded layout."""
    per_scale_loss = RecordingNaNLoss()
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[None, None],
    )
    multiscale_loss = MultiscaleLossWrapper(per_scale_loss=per_scale_loss, weights=[1.0, 1.0])
    pred, target, nan_mask = _nan_target_inputs()
    prepare = mocker.patch.object(
        multiscale_loss,
        "_prepare_for_smoothing",
        side_effect=lambda p, y, *_args: (p, y, [1, 1], [1, 1]),
    )
    mocker.patch(
        "anemoi.training.losses.multiscale.all_to_all_transpose",
        side_effect=lambda x, *_args: x,
    )

    multiscale_loss(_view(pred), _view(target), group=FakeGroup(size=2), grid_shard_sizes=[2, 2])

    smoothed_pred, smoothed_target = prepare.call_args.args[:2]
    torch.testing.assert_close(smoothed_pred, pred.masked_fill(nan_mask, 0.0))
    torch.testing.assert_close(smoothed_target, target.masked_fill(nan_mask, 0.0))
    assert len(per_scale_loss.targets) == 2
    for received in per_scale_loss.targets:
        assert torch.equal(torch.isnan(received), nan_mask)


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
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[provider],
    )
    loss = MultiscaleLossWrapper(per_scale_loss=MSELoss(), weights=[1.0])
    copy.deepcopy(loss)  # must not raise NotImplementedError on CSR tensors


@pytest.mark.parametrize("nan_variable", ["a", "f"], ids=["paired_target", "unpaired_forcing"])
def test_multiscale_masks_predictions_through_the_variable_mapping(mocker: MockerFixture, nan_variable: str) -> None:
    """NaN targets mask the prediction variables they are compared with, also when the layouts differ.

    Built from a config, the per-scale loss is wrapped in a LossVariableMapper: the prediction arrives in
    the model-output layout and the target in the full data layout, which also holds the forcings. The
    target's NaN mask therefore cannot be applied to the prediction position by position. A NaN in a
    target variable that is not compared (the forcing) must not mask any prediction.
    """
    mocker.patch(
        "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
        return_value=[_ring_smoother(), None],
    )
    # data layout: [f, a, b] with forcing "f"; model output layout: [a, b]
    data_indices = IndexCollection(DictConfig({"forcing": ["f"], "diagnostic": []}), {"f": 0, "a": 1, "b": 2})
    config = {
        "_target_": "anemoi.training.losses.MultiscaleLossWrapper",
        "weights": [1.0, 2.0],
        "loss_matrices": ["ring", None],
        "per_scale_loss": {"_target_": "anemoi.training.losses.CRPS", "scalers": [], "ignore_nans": True},
    }
    multiscale_loss = get_loss_function(DictConfig(config), scalers={}, data_indices=data_indices)
    pred, target, _ = _nan_target_inputs()  # two output variables, NaN in the first one ("a")
    forcing = torch.randn((*target.shape[:-1], 1), generator=torch.Generator().manual_seed(1))
    if nan_variable == "f":  # move the missing value from "a" to the forcing
        forcing[0, 0, 0, 1, 0] = float("nan")
        target = torch.nan_to_num(target, nan=0.5)
    target_full = torch.cat([forcing, target], dim=-1)

    loss = multiscale_loss(
        _view(pred),
        _view(target_full),
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    # Same loss on tensors that are aligned beforehand, without a variable mapping.
    aligned = MultiscaleLossWrapper(per_scale_loss=CRPS(ignore_nans=True), weights=[1.0, 2.0])
    expected = aligned(_view(pred), _view(target))
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)

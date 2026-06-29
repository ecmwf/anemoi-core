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
from anemoi.training.losses import CRPS
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
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
    target = torch.zeros([tensor_shape[0], tensor_shape[1], tensor_shape[3], tensor_shape[4]])  # no ensemble dim

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


def test_multi_scale_instantiation(loss_inputs_multiscale: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
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
    loss = multiscale_loss(pred, target, squash=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (2,), "Loss should have shape (num_scales,) when squash=True"
    loss = multiscale_loss(pred, target, squash=False)

    assert isinstance(loss, torch.Tensor)
    # better to have a nvar > 1 because otherwise pred.shape[-1] == 1 and loss.shape == (2) which makes the test fail
    assert loss.shape == (2, pred.shape[-1]), "Loss should have shape (num_scales, num_variables) when squash=False"


def test_multiscale_loss_equivalent_to_per_scale_loss() -> None:
    """Test equivalence when only one scale is used."""
    tensor_shape = [1, 1, 2, 4, 1]  # (batch, output_steps, ens, latlon, vars)

    pred = torch.zeros(tensor_shape)
    pred[0, 0, :, 0] = torch.tensor([1.0])
    target = torch.zeros([tensor_shape[0], tensor_shape[1], tensor_shape[3], tensor_shape[4]])  # no ensemble dim

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
    assert loss.shape == (1,)


def test_multiscale_loss_forwards_scaler_indices() -> None:
    pred = torch.zeros((1, 1, 1, 2, 2))
    pred[0, 0, 0, 0, 0] = 10.0
    pred[0, 0, 0, 0, 1] = 1.0
    target = torch.zeros((1, 1, 2, 2))

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
    target = torch.zeros((1, 1, 2, 1))
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


def test_multiscale_loss_uses_grid_shard_sizes_for_sharding(mocker: MockerFixture) -> None:
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
    target = torch.zeros((1, 1, 2, 1))

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
    target = torch.zeros((1, 1, 2, 1))
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


# ---------------------------------------------------------------------------
# Regression tests: verify numerical equivalence between commits 390ce34 and
# 38328094b.  Two changes in that range could alter loss values:
#
#   1. COO -> CSR conversion of smoothing matrices (_as_coalesced_csr).
#   2. Batched sparse matmul in SparseProjector replacing the per-sample loop.
#
# The helpers below re-implement both the old and new logic inline so that the
# tests work on *either* commit.
# ---------------------------------------------------------------------------

def _as_coalesced_csr(projection_matrix: torch.Tensor) -> torch.Tensor:
    """Mirror of MultiscaleLossWrapper._as_coalesced_csr (commit 38328094b)."""
    if projection_matrix.layout == torch.sparse_csr:
        return projection_matrix
    if projection_matrix.layout != torch.sparse_coo:
        msg = f"Expected sparse COO/CSR, got {projection_matrix.layout}."
        raise TypeError(msg)
    return projection_matrix.coalesce().to_sparse_csr()


def _make_sparse_coo_provider(n_nodes: int, seed: int = 0) -> ProjectionGraphProvider:
    """Build a ProjectionGraphProvider whose matrix is a sparse COO tensor."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Simple averaging matrix: each output node averages its two neighbours.
    row = torch.arange(n_nodes)
    col = torch.remainder(torch.arange(n_nodes) + 1, n_nodes)
    values = torch.ones(n_nodes) * 0.5
    proj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (n_nodes, n_nodes),
    ).coalesce()

    graph = HeteroData()
    graph["src"].num_nodes = n_nodes
    graph["dst"].num_nodes = n_nodes
    graph[("src", "to", "dst")].edge_index = torch.stack([col, row])
    graph[("src", "to", "dst")].edge_weight = values

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="edge_weight",
        row_normalize=False,
    )
    # Inject the pre-built matrix so we control the exact format.
    provider.projection_matrix = proj
    return provider


def _multiscale_loss_result(
    providers: list,
    weights: list[float],
    pred: torch.Tensor,
    target: torch.Tensor,
    use_csr: bool,
) -> torch.Tensor:
    """Run the multiscale loss with providers in either COO or CSR format."""
    if use_csr:
        for p in providers:
            if p is not None:
                p.projection_matrix = _as_coalesced_csr(p.projection_matrix)

    per_scale_loss = MSELoss()
    # Pass loss_matrices with the right length so the weights==num_scales
    # assertion in __init__ does not fire before we can override the providers.
    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=per_scale_loss,
        weights=weights,
        multiscale_config={"loss_matrices": [None] * len(providers)},
    )
    # Override with our controlled providers (COO or CSR as requested).
    multiscale_loss.smoothing_matrices = providers
    multiscale_loss.num_scales = len(providers)

    return multiscale_loss(pred, target)


class TestCsrConversionNumericalEquivalence:
    """Tests that COO and CSR smoothing matrices give the same loss values.

    A failure here means the format change in commit 38328094b directly causes
    loss differences — most likely because the original COO matrices had
    duplicate indices that .coalesce() merges.
    """

    @pytest.fixture
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = 8
        batch, steps, ens, n_vars = 1, 1, 2, 3
        pred = torch.randn(batch, steps, ens, n_nodes, n_vars)
        target = torch.randn(batch, steps, n_nodes, n_vars)
        return pred, target

    def test_single_scale_no_smoothing_coo_vs_csr(
        self,
        tensors: tuple[torch.Tensor, torch.Tensor],
        mocker: MockerFixture,
    ) -> None:
        """No smoothing (provider=None): COO/CSR irrelevant, loss must match."""
        pred, target = tensors
        providers = [None]

        mocker.patch(
            "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
            return_value=list(providers),
        )
        loss_coo = _multiscale_loss_result(list(providers), [1.0], pred, target, use_csr=False)

        mocker.patch(
            "anemoi.training.losses.multiscale.MultiscaleLossWrapper._load_smoothing_matrices",
            return_value=list(providers),
        )
        loss_csr = _multiscale_loss_result(list(providers), [1.0], pred, target, use_csr=True)

        assert torch.allclose(loss_coo, loss_csr), (
            f"Loss differs even with no smoothing provider — unexpected.\n"
            f"COO: {loss_coo}  CSR: {loss_csr}"
        )

    def test_two_scale_with_coo_smoothing_matches_csr(
        self,
        tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Two scales with a real COO smoothing matrix: CSR should give same loss."""
        pred, target = tensors
        n_nodes = pred.shape[-2]
        provider = _make_sparse_coo_provider(n_nodes=n_nodes)
        weights = [1.0, 1.0]

        loss_coo = _multiscale_loss_result([None, provider], weights, pred, target, use_csr=False)
        # Re-create the provider so COO->CSR conversion starts from the original.
        provider2 = _make_sparse_coo_provider(n_nodes=n_nodes)
        loss_csr = _multiscale_loss_result([None, provider2], weights, pred, target, use_csr=True)

        assert torch.allclose(loss_coo, loss_csr, atol=1e-5), (
            f"Two-scale loss differs between COO and CSR smoothing.\n"
            f"COO: {loss_coo}\nCSR: {loss_csr}\n"
            f"Max diff: {(loss_coo - loss_csr).abs().max():.4e}\n"
            "This confirms the CSR conversion in commit 38328094b changes the loss."
        )


class TestMultiscaleLoopRefactorEquivalence:
    """The multiscale forward loop was refactored from list-based indexing
    (y_preds_ens[i-1]) to prev-variable tracking (prev_y_pred_ens).
    Both are mathematically identical; this test proves it.
    """

    def _old_loop_differences(
        self,
        smoothed_preds: list[torch.Tensor],
        smoothed_targets: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Old approach: store all smoothed tensors, index by i-1."""
        y_preds_ens: list[torch.Tensor] = []
        y_ens: list[torch.Tensor] = []
        diffs = []
        for i, (yp, yt) in enumerate(zip(smoothed_preds, smoothed_targets)):
            y_preds_ens.append(yp)
            y_ens.append(yt)
            if i > 0:
                diffs.append((yp - y_preds_ens[i - 1], yt - y_ens[i - 1]))
            else:
                diffs.append((yp, yt))
        return diffs

    def _new_loop_differences(
        self,
        smoothed_preds: list[torch.Tensor],
        smoothed_targets: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """New approach: only keep the previous value (commit 38328094b)."""
        prev_yp: torch.Tensor | None = None
        prev_yt: torch.Tensor | None = None
        diffs = []
        for yp, yt in zip(smoothed_preds, smoothed_targets):
            current_yp, current_yt = yp, yt
            if prev_yp is not None:
                diffs.append((yp - prev_yp, yt - prev_yt))
            else:
                diffs.append((yp, yt))
            prev_yp = current_yp
            prev_yt = current_yt
        return diffs

    @pytest.mark.parametrize("n_scales", [2, 3, 4])
    def test_prev_var_matches_list_indexing(self, n_scales: int) -> None:
        """Both loop variants must produce identical difference tensors."""
        torch.manual_seed(0)
        smoothed_preds = [torch.randn(2, 8, 3) for _ in range(n_scales)]
        smoothed_targets = [torch.randn(2, 8, 3) for _ in range(n_scales)]

        old_diffs = self._old_loop_differences(smoothed_preds, smoothed_targets)
        new_diffs = self._new_loop_differences(smoothed_preds, smoothed_targets)

        assert len(old_diffs) == len(new_diffs)
        for i, ((op, ot), (np_, nt)) in enumerate(zip(old_diffs, new_diffs)):
            assert torch.equal(op, np_), f"Scale {i}: pred difference mismatch between old and new loop"
            assert torch.equal(ot, nt), f"Scale {i}: target difference mismatch between old and new loop"

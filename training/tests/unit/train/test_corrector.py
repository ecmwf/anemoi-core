# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.train.methods.corrector import CorrectorGNN
from anemoi.training.train.methods.corrector import CorrectorMLP
from anemoi.training.train.methods.corrector import InstrumentCorrectorMLPs


def test_corrector_mlp_zero_initialised() -> None:
    mlp = CorrectorMLP(n_target=3, n_corrector=2, hidden_dim=8)
    y = torch.randn(2, 4, 3)
    corr = torch.randn(2, 4, 2)
    # zero-init output head -> no correction at initialisation
    assert torch.allclose(mlp(y, corr), torch.zeros_like(y))


def test_corrector_gnn_zero_initialised_and_shape() -> None:
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 2)
    gnn = CorrectorGNN(
        n_target=3,
        n_corrector=1,
        hidden_dim=8,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        num_layers=2,
    )
    y = torch.randn(2, num_nodes, 3)
    corr = torch.randn(2, num_nodes, 1)
    out = gnn(y, corr)
    assert out.shape == (2, num_nodes, 3)
    assert torch.allclose(out, torch.zeros_like(out))  # zero-init head


def test_instrument_corrector_mlp_additive_and_targeted() -> None:
    # two output channels named hirs_1, hirs_2 plus an unrelated z; corrector var geom
    output_name_to_index = {"z": 0, "hirs_1": 1, "hirs_2": 2}
    corrector = InstrumentCorrectorMLPs(
        instrument_groups={"hirs": {"corrector_variables": ["geom"], "channels": None}},
        all_corrector_names=["geom"],
        output_name_to_index=output_name_to_index,
        hidden_dim=8,
        corrector_type="mlp",
    )
    y_pred = torch.randn(2, 5, 3)
    corrector_vars = torch.randn(2, 5, 1)
    out = corrector(y_pred, corrector_vars)
    # zero-init -> identity at start, but the non-targeted channel must always pass through
    assert torch.allclose(out[..., 0], y_pred[..., 0])
    assert torch.allclose(out, y_pred)  # zero-init correction

    # after perturbing the output head, only hirs channels change
    corrector.mlps["hirs"].out.weight.data.fill_(0.1)
    corrector.mlps["hirs"].out.bias.data.fill_(0.5)
    out2 = corrector(y_pred, corrector_vars)
    assert torch.allclose(out2[..., 0], y_pred[..., 0])  # z untouched
    assert not torch.allclose(out2[..., 1], y_pred[..., 1])  # hirs_1 corrected


def test_instrument_corrector_prefix_matching() -> None:
    output_name_to_index = {"mwt_1": 0, "mwt_2": 1, "other": 2}
    corrector = InstrumentCorrectorMLPs(
        instrument_groups={"mwt": {"corrector_variables": ["v"], "channels": None}},
        all_corrector_names=["v", "unused"],
        output_name_to_index=output_name_to_index,
        hidden_dim=4,
        corrector_type="mlp",
    )
    target_idx = corrector._target_idx_mwt.tolist()
    assert target_idx == [0, 1]  # mwt_1, mwt_2 matched by prefix, not "other"

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.input_transform import BaseInputTransform
from anemoi.models.variable.variabletokenizer import MeanPoolingTransform
from anemoi.models.variable.variabletokenizer import MultiHeadAttentionTransform
from anemoi.models.variable.variabletokenizer import SumPoolingTransform

DATASET_METADATA = {
    "data": {
        "variables_metadata": {
            "2t": {"mars": {"levtype": "sfc", "param": "2t", "time": 0, "step": 0}},
            "t_850": {"mars": {"levtype": "pl", "levelist": 850, "param": "t", "time": 0, "step": 0}},
            "tp": {"mars": {"levtype": "sfc", "param": "tp", "time": 1800, "step": 6}},
            "cos_latitude": {"computed_forcing": True, "constant_in_time": True},
        }
    }
}
VARIABLES = ["2t", "t_850", "tp", "cos_latitude"]


@pytest.mark.parametrize(
    "tokenizer_class, kwargs",
    [
        (MultiHeadAttentionTransform, {"num_heads": 2, "chunk_size": 3}),
        (MeanPoolingTransform, {}),
        (SumPoolingTransform, {}),
    ],
)
def test_tokenizer_returns_encoder_input(tokenizer_class, kwargs):
    batch, time, ensemble, grid, n_attributes, out_dim = 2, 3, 1, 5, 4, 6
    tokenizer = tokenizer_class(emb_dim=8, out_dim=out_dim, dataset_metadata=DATASET_METADATA, **kwargs)
    x = torch.randn(batch, time, ensemble, grid, len(VARIABLES))
    node_attributes = torch.randn(batch * ensemble * grid, n_attributes)

    out = tokenizer(x, node_attributes, variables=VARIABLES)

    assert isinstance(tokenizer, BaseInputTransform)
    assert tokenizer.output_dim == out_dim
    assert out.shape == (batch * ensemble * grid, time * out_dim + n_attributes)
    torch.testing.assert_close(out[:, -n_attributes:], node_attributes)


def test_tokenizer_accepts_a_subset_of_variables():
    tokenizer = MultiHeadAttentionTransform(emb_dim=8, out_dim=6, dataset_metadata=DATASET_METADATA, num_heads=2)
    x = torch.randn(1, 1, 1, 5, 2)

    out = tokenizer(x, torch.zeros(5, 0), variables=["tp", "2t"])

    assert out.shape == (5, 6)


def test_tokenizer_needs_metadata():
    with pytest.raises(ValueError, match="metadata"):
        MultiHeadAttentionTransform(emb_dim=8, out_dim=6, num_heads=2)

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
import torch.nn as nn
from anemoi.training.utils.checkpoint import transfer_linear_layer

@pytest.mark.parametrize(
    ("old_leads", "new_leads"),
    [
        ([6], [3,6]),
        ([6], [1,2,3,4,5,6]),
        ([6], [1,2,3]),
        ([1,2,3], [6]),
    ],
    ids=[
        "test_transfer_outputs_3houry",
        "test_transfer_outputs_hourly",
        "test_transfer_outputs_no_commons",
        "test_transfer_outputs_no_commons_inverse",
    ],
)
def test_transfer_outputs(old_leads: list[int], new_leads: list[int]) -> None:
    data_dim = 3
    latent_dim = 6
    sample_x = torch.randn(1,latent_dim)

    old_linear = nn.Linear(latent_dim, data_dim * len(old_leads))

    old_out = old_linear(sample_x)

    new_linear = transfer_linear_layer(lin_layer=old_linear,
                                            old_leads=old_leads,
                                            new_leads=new_leads,
                                            transfer_on_output=True)

    # assert that the output size of the new layer is num_leads * data_dim
    assert new_linear.weight.shape[0] == data_dim * len(new_leads)

    new_out = new_linear(sample_x)

    common_lead_times = [lead for lead in old_leads if lead in new_leads]

    for lead in common_lead_times:
        old_index = old_leads.index(lead)
        new_index = new_leads.index(lead)

        # assert that the parameter blocks relative to the common lead time are the same
        assert (old_linear.weight[old_index*data_dim:(old_index+1)*data_dim]
                == new_linear.weight[new_index*data_dim:(new_index+1)*data_dim]).all()

        # assert that the output for the common lead time is the same up to floating point precision
        torch.testing.assert_close(
            old_out[:,old_index*data_dim:(old_index+1)*data_dim],
            new_out[:,new_index*data_dim:(new_index+1)*data_dim],
            rtol=1e-5,
            atol=1e-8,
        )

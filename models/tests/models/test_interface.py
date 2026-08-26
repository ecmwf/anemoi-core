# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import Mock

import torch

from anemoi.models.interface import AnemoiModelInterface


def test_predict_step_forwards_gather_out() -> None:
    interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(interface)
    interface.model = Mock()
    interface.pre_processors = Mock()
    interface.post_processors = Mock()
    interface.n_step_input = 2

    interface.predict_step({"data": torch.empty(1, 2, 3, 4)}, gather_out=False)

    assert interface.model.predict_step.call_args.kwargs["gather_out"] is False

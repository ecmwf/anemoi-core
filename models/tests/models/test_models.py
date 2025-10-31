# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.models.obsinterpolator import AnemoiModelObsInterpolator


def test_obsinterpolator_calculate_input_dim():
    # Construct a lightweight instance without running full __init__
    m = AnemoiModelObsInterpolator.__new__(AnemoiModelObsInterpolator)
    m.multi_step = 6  
    m.num_input_channels = 4  # e.g. 4 observation variables
    m.known_future_variables = ["U_10M_NWP", "V_10M_NWP", "TD_2M_NWP", "T_2M_NWP", "TOT_PREC_NWP"]  # len = 5
    class _NA:
        attr_ndims = {"data": 7} # graph attributes

    m.node_attributes = _NA()
    m._graph_name_data = "data"

    # Expected:
    # multi_step * num_input_channels
    # + 2 * len(known_future_variables)    (boundary future + intermediate future)
    # + 1                                  (relative_frac_interval)
    # + node_attributes.attr_ndims[data]
    expected = (6 * 4) + (2 * 5) + 1 + 7  # 24 + 10 + 1 + 7 = 42

    assert m._calculate_input_dim() == expected


def test_obsinterpolator_calculate_input_dim_no_future_vars():
    m = AnemoiModelObsInterpolator.__new__(AnemoiModelObsInterpolator)
    m.multi_step = 2
    m.num_input_channels = 3
    m.known_future_variables = []

    class _NA:
        attr_ndims = {"data": 2}

    m.node_attributes = _NA()
    m._graph_name_data = "data"

    expected = (2 * 3) + (2 * 0) + 1 + 2  # 6 + 0 + 1 + 2 = 9
    assert m._calculate_input_dim() == expected


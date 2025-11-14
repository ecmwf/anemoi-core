# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.models.interpolator import AnemoiModelEncProcDecInterpolator


def test_resolve_mass_conservations_conserves_total():
    # Dimensions
    B, T, E, G = 3, 4, 2, 5  # batch, time, ensemble, grid
    V = 6  # number of accumulated variables (targets), var0..var5

    # Create dummy "self" with required attributes used by resolve_mass_conservations
    class _NI:
        def __init__(self, mapping: dict[str, int]) -> None:
            self.name_to_index = mapping

    class _ModelIdx:
        def __init__(self, out_map: dict[str, int], in_map: dict[str, int]) -> None:
            self.output = _NI(out_map)
            self.input = _NI(in_map)

    class _DI:
        def __init__(self, out_map: dict[str, int], in_map: dict[str, int]) -> None:
            self.model = _ModelIdx(out_map, in_map)

    dummy = type("Dummy", (), {})()
    # All output channels are mass-conserving targets; constraints come from inputs at the same indices
    dummy.map_accum_indices = {
        "target_idxs": torch.arange(V, dtype=torch.long),
        "constraint_idxs": torch.arange(V, dtype=torch.long),
    }
    # Provide output/input name_to_index maps (not used when all are targets)
    out_map = {f"var{i}": i for i in range(V)}
    in_map = {f"var{i}": i for i in range(V)}
    dummy.data_indices = _DI(out_map, in_map)

    # y_preds logits (values act as logits for softmax-based time weights)
    y_preds = torch.zeros((B, T, E, G, V), dtype=torch.float32)

    # x_input containing the right-boundary constraints at the last time step
    x_input = torch.zeros((B, T, E, G, V), dtype=torch.float32)

    # Vectorized totals: var0 is always 0; var1..var5 are in [0, 4.4)
    # Accum variables are usually positive and normalized using std (no mean removal),
    # so their normalized values should range between 0 and around 4.4 excluding extreme tail events.
    torch.manual_seed(0)
    totals = torch.zeros(V, dtype=torch.float32)
    totals[1:] = torch.rand(V - 1, dtype=torch.float32) * 4.4

    # Broadcast totals to the right boundary across batch/ensemble/grid dims
    x_input[:, -1:, ..., :] = totals

    # Call as unbound method on the class, passing the dummy self
    y_out = AnemoiModelEncProcDecInterpolator.resolve_mass_conservations(
        dummy, y_preds.clone(), x_input, include_right_boundary=True
    )

    # Expect: time dimension increased by 1
    assert y_out.shape[1] == T + 1

    # Sum across time for all target channels equals the right-boundary constraints exactly
    target_idxs = dummy.map_accum_indices["target_idxs"]
    constraint_idxs = dummy.map_accum_indices["constraint_idxs"]
    summed = y_out[..., target_idxs].sum(dim=1, keepdim=True)  # (B, 1, E, G, V)
    constraints = x_input[:, -1:, ..., constraint_idxs]
    assert torch.allclose(summed, constraints, atol=1e-6, rtol=0), "Mass is not conserved across time steps"


def test_setup_mass_conserving_accumulations_builds_indices():
    # Build dummy data_indices and config with a valid mapping
    class _NI:
        def __init__(self, mapping):
            self.name_to_index = mapping

    class _ModelIdx:
        def __init__(self, out_map, in_map):
            self.output = _NI(out_map)
            self.input = _NI(in_map)

    class _Data:
        def __init__(self, forcing):
            self._forcing = forcing

    class _DI:
        def __init__(self, out_map, in_map, forcing):
            self.model = _ModelIdx(out_map, in_map)
            self.data = _Data(forcing)

    class _CfgModel:
        def __init__(self, mapping):
            self.mass_conserving_accumulations = mapping

    class _Cfg:
        def __init__(self, mapping):
            self.model = _CfgModel(mapping)

    out_map = {"outA": 0, "outB": 1, "outC": 2}
    in_map = {"inA": 10, "inB": 11, "inC": 12}
    forcing = {"inA", "inB", "inC"}

    data_indices = _DI(out_map, in_map, forcing)
    mapping = {"outA": "inA", "outC": "inC"}
    cfg = _Cfg(mapping)

    dummy = type("Dummy", (), {})()

    AnemoiModelEncProcDecInterpolator.setup_mass_conserving_accumulations(dummy, data_indices, cfg)

    assert isinstance(dummy.map_accum_indices, torch.nn.ParameterDict)
    t = dummy.map_accum_indices["target_idxs"].detach().cpu().tolist()
    c = dummy.map_accum_indices["constraint_idxs"].detach().cpu().tolist()
    assert dummy.map_accum_indices["target_idxs"].dtype == torch.long
    assert dummy.map_accum_indices["constraint_idxs"].dtype == torch.long
    assert t == [0, 2]
    assert c == [10, 12]


def test_setup_mass_conserving_accumulations_none_mapping_sets_none():
    # Config without 'mass_conserving_accumulations' should set map_accum_indices to None
    class _CfgModelEmpty:
        pass

    class _Cfg:
        def __init__(self):
            self.model = _CfgModelEmpty()

    dummy = type("Dummy", (), {})()
    # data_indices won't be used in this branch; pass any object
    AnemoiModelEncProcDecInterpolator.setup_mass_conserving_accumulations(dummy, object(), _Cfg())
    assert dummy.map_accum_indices is None


def test_setup_mass_conserving_accumulations_raises_on_missing_names():
    # Prepare common scaffolding
    class _NI:
        def __init__(self, mapping):
            self.name_to_index = mapping

    class _ModelIdx:
        def __init__(self, out_map, in_map):
            self.output = _NI(out_map)
            self.input = _NI(in_map)

    class _Data:
        def __init__(self, forcing):
            self._forcing = forcing

    class _DI:
        def __init__(self, out_map, in_map, forcing):
            self.model = _ModelIdx(out_map, in_map)
            self.data = _Data(forcing)

    class _CfgModel:
        def __init__(self, mapping):
            self.mass_conserving_accumulations = mapping

    class _Cfg:
        def __init__(self, mapping):
            self.model = _CfgModel(mapping)

    out_map = {"outA": 0, "outB": 1}
    in_map = {"inA": 10, "inB": 11, "inZ": 15}
    forcing = {"inA", "inB"}  # 'inZ' is intentionally missing

    # Case 1: input constraint not in forcing set
    data_indices = _DI(out_map, in_map, forcing)
    cfg_bad_input = _Cfg({"outA": "inZ"})
    dummy1 = type("Dummy", (), {})()
    with pytest.raises(AssertionError):
        AnemoiModelEncProcDecInterpolator.setup_mass_conserving_accumulations(dummy1, data_indices, cfg_bad_input)

    # Case 2: output variable not in model output mapping
    cfg_bad_output = _Cfg({"outZ": "inA"})
    dummy2 = type("Dummy", (), {})()
    with pytest.raises(AssertionError):
        AnemoiModelEncProcDecInterpolator.setup_mass_conserving_accumulations(dummy2, data_indices, cfg_bad_output)

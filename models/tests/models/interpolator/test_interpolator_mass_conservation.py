# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Real-instantiation tests for the asymmetric dataset roles of the base transport model.

A conditioning-only dataset is encoded (its latent joins the additive merge) but is never
predicted: it has no decoder, no noised (corrupted) target concatenated to its encoder input,
and never appears in the forward output dict. The feature is general to the transport family and
is driven purely by the explicit ``model.model.conditioning_only_datasets`` config list.

The dual role, exercised in the second half of this module, is a history-less dataset: predicted
(decoder, noised target, present in the output) but supplying NO input history, so its encoder
input is the noised target plus node attributes only. It is declared in
``model.model.history_less_datasets``. Together the two off-diagonal roles let the honest
downscaling training shape be ``x = {source}`` only, with the high-resolution target entering the
network solely through its noised target in ``conditioned_target``.

These tests construct :class:`AnemoiTransportModelEncProcDec` through its real ``__init__`` with a
real graph and real graph-transformer encoders/decoders/processor, with two datasets:

- ``aux``  — conditioning-only (its own small grid), and
- ``output`` — predicted.

Design note verified here: an input-only dataset still carries a non-empty ``model.output``
(``aux``'s prognostic ``u, v``), so conditioning-only cannot be inferred from ``data_indices`` — it
must be declared explicitly. That is exactly why the config list is the source of truth.

Construction boilerplate (graph, layer kernels, model_config skeleton, index collections,
statistics) is shared with ``test_transport_residual_forward.py`` via the factory fixtures in
``conftest.py``; only the ``build_model`` fixture below is specific to this module's two-dataset
(``aux``, ``output``) shape.
"""

import pytest
import torch

from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec

# ── grid sizes ────────────────────────────────────────────────────────────────
N_AUX = 5  # conditioning-only dataset grid
N_OUTPUT = 8  # predicted dataset grid (different size on purpose)
N_HIDDEN = 4  # tiny hidden mesh

NUM_CHANNELS = 16
NOISE_CHANNELS = 8
NOISE_COND_DIM = 4
N_STEP_INPUT = 2
N_STEP_OUTPUT = 1

# ── variable roles ─────────────────────────────────────────────────────────────
# aux "conditioning-only": u(0) v(1) sforce(2), forcing=[sforce] -> prognostic=[u, v]
#   model.input = [u, v, sforce] (3), model.output = [u, v] (2) -> NON-EMPTY output even though
#   this dataset is only ever an input. This is what makes explicit config necessary.
# output "predicted": u(0) v(1) force(2), forcing=[force] -> prognostic=[u, v]
#   model.input = [u, v, force] (3), model.output = [u, v] (2)
_AUX_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}
_OUTPUT_NAME_TO_INDEX = {"u": 0, "v": 1, "force": 2}


@pytest.fixture
def build_model(index_collection_factory, statistics_factory, tiny_graph_factory, transport_model_config_factory):
    """Build a real two-dataset (``aux``, ``output``) :class:`AnemoiTransportModelEncProcDec`."""

    def _build(
        conditioning_only_datasets: list[str],
        history_less_datasets: list[str] | None = None,
    ) -> AnemoiTransportModelEncProcDec:
        data_indices = {
            "aux": index_collection_factory(_AUX_NAME_TO_INDEX, forcing=["sforce"]),
            "output": index_collection_factory(_OUTPUT_NAME_TO_INDEX, forcing=["force"]),
        }
        statistics = {
            "aux": statistics_factory(len(_AUX_NAME_TO_INDEX)),
            "output": statistics_factory(len(_OUTPUT_NAME_TO_INDEX)),
        }
        # aux has NO hidden->aux decoder edges on purpose: conditioning-only datasets are never decoded.
        graph = tiny_graph_factory(
            {"aux": N_AUX, "output": N_OUTPUT, "hidden": N_HIDDEN},
            [("aux", "hidden"), ("output", "hidden"), ("hidden", "output"), ("hidden", "hidden")],
        )
        model_config = transport_model_config_factory(
            num_channels=NUM_CHANNELS,
            noise_channels=NOISE_CHANNELS,
            noise_cond_dim=NOISE_COND_DIM,
            model_extra={
                "conditioning_only_datasets": list(conditioning_only_datasets),
                "history_less_datasets": list(history_less_datasets or []),
            },
        )
        return AnemoiTransportModelEncProcDec(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            n_step_input=N_STEP_INPUT,
            n_step_output=N_STEP_OUTPUT,
            graph_data=graph,
        )

    return _build


@pytest.fixture
def model(build_model) -> AnemoiTransportModelEncProcDec:
    return build_model(["aux"])


def _expected_input_dim(model: AnemoiTransportModelEncProcDec, dataset: str, conditioning_only: bool) -> int:
    """Hand formula for the encoder input dim.

    history = n_step_input * len(model.input) + node-attribute dims; predicted datasets add a
    corrupted-target term n_step_output * len(model.output); conditioning-only datasets do not.
    """
    history = N_STEP_INPUT * len(model.data_indices[dataset].model.input) + model.node_attributes.attr_ndims[dataset]
    if conditioning_only:
        return history
    return history + N_STEP_OUTPUT * len(model.data_indices[dataset].model.output)


# ── test 1: construction, decoder membership, encoder input dims ───────────────
def test_construction_decoder_membership_and_input_dims(model: AnemoiTransportModelEncProcDec) -> None:
    # The conditioning-only design cannot be inferred from data_indices: aux still has model.output.
    assert len(model.data_indices["aux"].model.output) == 2  # u, v — non-empty despite input-only role
    assert model.conditioning_only_datasets == {"aux"}

    # aux is encoded but never decoded; output is decoded.
    assert "aux" in model.encoder
    assert "output" in model.encoder
    assert "aux" not in model.decoder
    assert "output" in model.decoder
    assert "aux" not in model.decoder_graph_provider
    assert "output" in model.decoder_graph_provider

    # aux input dim has NO corrupted-target term; output input dim is unchanged (has the term).
    expected_aux = _expected_input_dim(model, "aux", conditioning_only=True)
    expected_output = _expected_input_dim(model, "output", conditioning_only=False)
    assert model.input_dim["aux"] == expected_aux
    assert model.input_dim["output"] == expected_output

    # The corrupted-target term is exactly what separates the two: aux lacks it, output has it.
    aux_history = N_STEP_INPUT * len(model.data_indices["aux"].model.input) + model.node_attributes.attr_ndims["aux"]
    assert model.input_dim["aux"] == aux_history  # no + n_step_output * num_output_channels
    output_corrupted_term = N_STEP_OUTPUT * len(model.data_indices["output"].model.output)
    output_history = (
        N_STEP_INPUT * len(model.data_indices["output"].model.input) + model.node_attributes.attr_ndims["output"]
    )
    assert model.input_dim["output"] == output_history + output_corrupted_term

    # The real encoders were built with those dims (only a real instantiation can check this).
    assert model.encoder["aux"].in_channels_src == expected_aux
    assert model.encoder["output"].in_channels_src == expected_output


def _forward_inputs(model: AnemoiTransportModelEncProcDec, batch_size: int = 2) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(11)
    n_in_aux = len(model.data_indices["aux"].model.input)  # 3
    n_in_output = len(model.data_indices["output"].model.input)  # 3
    n_out_output = len(model.data_indices["output"].model.output)  # 2
    x = {
        "aux": torch.randn(batch_size, N_STEP_INPUT, 1, N_AUX, n_in_aux, generator=generator),
        "output": torch.randn(batch_size, N_STEP_INPUT, 1, N_OUTPUT, n_in_output, generator=generator),
    }
    # conditioned_target and condition contain ONLY the predicted dataset (mirrors the objective).
    conditioned_target = {
        "output": torch.randn(batch_size, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output, generator=generator),
    }
    condition = {"output": torch.full((batch_size, 1, 1, 1, 1), 0.5)}
    return x, conditioned_target, condition


# ── test 2: full forward returns only the decoded dataset ──────────────────────
def test_forward_returns_only_predicted_dataset(model: AnemoiTransportModelEncProcDec) -> None:
    x, conditioned_target, condition = _forward_inputs(model)
    n_out_output = len(model.data_indices["output"].model.output)

    out = model.forward(x, conditioned_target, condition)

    assert set(out.keys()) == {"output"}  # aux is absent from the result
    assert out["output"].shape == (2, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output)
    assert torch.isfinite(out["output"]).all()


# ── test 3: a predicted dataset missing its conditioned target still raises loudly ──
def test_missing_conditioned_target_for_predicted_dataset_raises(model: AnemoiTransportModelEncProcDec) -> None:
    x, _, condition = _forward_inputs(model)

    # "output" is NOT conditioning-only, so a missing corrupted target must raise (loud contract).
    with pytest.raises(AssertionError, match="not conditioning-only"):
        model._forward_transport_network(x, {}, condition)


# ── test 4: an unknown conditioning-only name raises at construction ───────────
def test_unknown_conditioning_only_dataset_raises_at_construction(build_model) -> None:
    with pytest.raises(ValueError, match="conditioning_only_datasets references unknown datasets"):
        build_model(["ghost"])


# ── asymmetric roles: history-less predicted datasets (the dual of conditioning-only) ──
# aux stays conditioning-only (in x, not in conditioned_target); output becomes history-less (in
# conditioned_target, absent from x). This exercises both off-diagonal cells of the 2x2 role matrix
# at once: the honest training shape is ``x = {aux}`` only.


@pytest.fixture
def asymmetric_model(build_model) -> AnemoiTransportModelEncProcDec:
    return build_model(["aux"], ["output"])


def _asymmetric_forward_inputs(model: AnemoiTransportModelEncProcDec, batch_size: int = 2) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(11)
    n_in_aux = len(model.data_indices["aux"].model.input)  # 3
    n_out_output = len(model.data_indices["output"].model.output)  # 2
    # Honest shape: x carries ONLY the conditioning-only source; the history-less target is absent.
    x = {"aux": torch.randn(batch_size, N_STEP_INPUT, 1, N_AUX, n_in_aux, generator=generator)}
    conditioned_target = {
        "output": torch.randn(batch_size, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output, generator=generator),
    }
    condition = {"output": torch.full((batch_size, 1, 1, 1, 1), 0.5)}
    return x, conditioned_target, condition


def test_history_less_construction_membership_and_input_dims(asymmetric_model) -> None:
    model = asymmetric_model
    assert model.conditioning_only_datasets == {"aux"}
    assert model.history_less_datasets == {"output"}

    # aux is encoded, never decoded; output is BOTH encoded and decoded (it is predicted).
    assert "aux" in model.encoder and "output" in model.encoder
    assert "aux" not in model.decoder and "output" in model.decoder

    # aux encoder input dim: input history + node attrs (conditioning-only, no corrupted-target term).
    aux_history = N_STEP_INPUT * len(model.data_indices["aux"].model.input) + model.node_attributes.attr_ndims["aux"]
    assert model.input_dim["aux"] == aux_history

    # output encoder input dim (history-less): corrupted-target term + node attrs, NO input-history term.
    output_corrupted = N_STEP_OUTPUT * len(model.data_indices["output"].model.output)
    output_node_attr = model.node_attributes.attr_ndims["output"]
    expected_output = output_node_attr + output_corrupted
    assert model.input_dim["output"] == expected_output
    assert model.encoder["output"].in_channels_src == expected_output

    # Contrast: an ordinary predicted dataset would ALSO carry the input-history term, so the
    # history-less encoder is strictly smaller by exactly that omitted term.
    output_history_term = N_STEP_INPUT * len(model.data_indices["output"].model.input)
    assert output_history_term > 0
    assert model.input_dim["output"] == expected_output  # the history term is genuinely absent
    assert model.input_dim["output"] + output_history_term > model.input_dim["output"]


def test_history_less_forward_returns_only_predicted_dataset(asymmetric_model) -> None:
    model = asymmetric_model
    x, conditioned_target, condition = _asymmetric_forward_inputs(model)
    n_out_output = len(model.data_indices["output"].model.output)

    out = model.forward(x, conditioned_target, condition)

    assert set(x.keys()) == {"aux"}  # honest shape: the history-less target is not a model input
    assert set(out.keys()) == {"output"}
    assert out["output"].shape == (2, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output)
    assert torch.isfinite(out["output"]).all()


def test_history_less_declared_but_present_in_x_raises(asymmetric_model) -> None:
    model = asymmetric_model
    x, conditioned_target, condition = _asymmetric_forward_inputs(model)
    # Put the declared-history-less target back into x: a config contract violation.
    n_in_output = len(model.data_indices["output"].model.input)
    x = {**x, "output": torch.randn(2, N_STEP_INPUT, 1, N_OUTPUT, n_in_output)}
    with pytest.raises(ValueError, match="history_less_datasets"):
        model._forward_transport_network(x, conditioned_target, condition)


def test_history_less_present_but_undeclared_raises(build_model) -> None:
    # Build with output NOT declared history-less (so it is sized as an ordinary predicted dataset),
    # then run the honest x = {aux}-only forward: the derived history-less role has no declaration.
    model = build_model(["aux"], history_less_datasets=[])
    x, conditioned_target, condition = _asymmetric_forward_inputs(model)
    with pytest.raises(ValueError, match="history_less_datasets"):
        model._forward_transport_network(x, conditioned_target, condition)


def test_unknown_history_less_dataset_raises_at_construction(build_model) -> None:
    with pytest.raises(ValueError, match="history_less_datasets references unknown datasets"):
        build_model(["aux"], ["ghost"])


def test_dataset_cannot_be_both_conditioning_only_and_history_less(build_model) -> None:
    with pytest.raises(ValueError, match="cannot be both conditioning-only and history-less"):
        build_model(["aux"], ["aux"])

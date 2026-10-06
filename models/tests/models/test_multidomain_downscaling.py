# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

"""Scientific routing tests; CPU mapper is a stand-in, not GPU qualification."""

import pytest
import torch

from anemoi.models.models.multidomain_decoder import (
    PaperMultidomainDecoder,
    SaturationHumidityCoordinate,
)
from anemoi.models.models.multidomain_transport import _PaperSinusoidalEmbeddings


class SmallMapper(torch.nn.Module):
    def __init__(self, **configuration):
        super().__init__()
        self.extract = torch.nn.Linear(configuration["in_channels_src"], configuration["out_channels_dst"], bias=False)

    def forward(self, values, **configuration):
        source, _ = values
        return self.extract(source[configuration["edge_index"][0]])


@pytest.fixture(params=[(2, 2), (4,)])
def decoder_case(monkeypatch, request):
    import anemoi.models.layers.mapper

    monkeypatch.setattr(anemoi.models.layers.mapper, "GraphTransformerBackwardMapper", SmallMapper)
    torch.manual_seed(20261006)
    model = PaperMultidomainDecoder(
        factors=2,
        static_features=3,
        union_channels=4,
        domain_native_points={"MEPS": 4},
        pools=1,
        humidity_union_indices=(0,),
        width=8,
        heads=2,
        chunks=1,
        learned_edge_channels=2,
        local_width=4,
        local_dilations=(1,),
        precipitation_width=4,
    ).eval()
    # Initial zero heads alone cannot detect a broken specialization route.
    with torch.no_grad():
        for module in (
            model.precipitation_lift[-1],
            model.precipitation_head[-1],
            model.humidity_saturation_head[-1],
            model.local_grid_refiner.output_projection,
            model.local_grid_precipitation_refiner.output_projection,
        ):
            module.weight.normal_(std=0.15)
            module.bias.fill_(0.03)
    inputs = dict(
        factor_mesh=torch.randn(2, 2),
        context_native_union=torch.randn(4, 4),
        availability_union=torch.ones(4),
        mesh_static=torch.randn(2, 3),
        native_static=torch.randn(4, 3),
        linear_skip_union=torch.randn(4, 4),
        precipitation_mesh=torch.tensor([[0.3], [0.9]]),
        precipitation_direct_skip=torch.randn(4, 1),
        edge_index=torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]),
        physical_edge_attributes=torch.randn(4, 3),
        domain="MEPS",
        pool_index=0,
        lead_hours=0.0,
        cadence_hours=6.0,
        grid_shape=request.param,
        humidity_direct_skip=torch.randn(4, 1),
    )
    return model, inputs


def test_checkpoint_keys_and_shapes_load_strictly(decoder_case):
    model, _ = decoder_case
    model.load_state_dict(model.state_dict(), strict=True)
    assert "factor_lift.0.weight" in model.state_dict()
    assert "local_grid_precipitation_refiner.output_projection.weight" in model.state_dict()


@pytest.mark.parametrize("stage", ["humidity", "joint"])
def test_rain_condition_never_changes_frozen_atmosphere_or_humidity(decoder_case, stage):
    model, inputs = decoder_case
    local = model(**inputs, decoder_stage="local")
    before = model(**inputs, decoder_stage=stage)
    after = model(**dict(inputs, precipitation_mesh=inputs["precipitation_mesh"] + 4), decoder_stage=stage)
    for actual in (before, after):
        torch.testing.assert_close(actual["atmospheric_residual"], local["atmospheric_residual"], rtol=0, atol=0)
    torch.testing.assert_close(
        before["humidity_saturation_coordinate"], after["humidity_saturation_coordinate"], rtol=0, atol=0
    )
    if stage == "joint":
        assert not torch.equal(before["precipitation_coordinate"], after["precipitation_coordinate"])


def test_rain_stage_and_joint_publish_identical_rain(decoder_case):
    model, inputs = decoder_case
    rain = model(**inputs, decoder_stage="precipitation")["precipitation_coordinate"]
    joint = model(**inputs, decoder_stage="joint")["precipitation_coordinate"]
    torch.testing.assert_close(rain, joint, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["bounded_logit", "softplus_ratio"])
def test_humidity_coordinate_roundtrip(kind):
    coordinate = SaturationHumidityCoordinate([850, 500], [1.1, 1.2], [0.001, 0.001], coordinate_kind=kind)
    temperature = torch.tensor([[280.0, 260.0]])
    latent = torch.tensor([[-1.0, 0.3]])
    physical = coordinate.inverse(latent, temperature)
    torch.testing.assert_close(coordinate.coordinate(physical, temperature), latent, rtol=1e-5, atol=1e-6)


def test_flow_time_embedding_matches_recovered_formula():
    embedding = _PaperSinusoidalEmbeddings()
    time = torch.tensor([[0.4]])
    angles = time * torch.exp(-torch.log(torch.tensor(1000.0)) * torch.arange(16) / 16)
    torch.testing.assert_close(embedding(time), torch.cat((angles.sin(), angles.cos()), -1), rtol=0, atol=0)
    with pytest.raises(ValueError, match="even"):
        _PaperSinusoidalEmbeddings(31)

# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import math

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.spatial_dropout import RandomSpatialDropout
from anemoi.models.preprocessing.structured_dropout import StructuredObsDropout
from anemoi.models.schemas.data_processor import RandomSpatialDropoutSchema
from anemoi.models.schemas.data_processor import StructuredObsDropoutSchema

NAMES = [
    "z_500",
    "t_500",
    "t_850",
    "mwt_1",
    "mwt_2",
    "mwt_cos_sza",
    "gpsro_8000",
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "lsm",
]
NAME_TO_INDEX = {n: i for i, n in enumerate(NAMES)}
SONDE = [0, 1, 2]
MWT = [3, 4, 5]
GPSRO = [6]
FORCING = [7, 8, 9, 10, 11]


@pytest.fixture()
def data_indices() -> IndexCollection:
    config = DictConfig(
        {
            "forcing": ["cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude", "lsm"],
            "diagnostic": [],
            "corrector": ["mwt_cos_sza"],
        },
    )
    return IndexCollection(data_config=config, name_to_index=NAME_TO_INDEX)


def _groups(**overrides) -> dict:
    groups = {
        "radiosonde": {"variables": ["z_*", "t_*"]},
        "mw_temp": {"variables": ["mwt_*"]},
        "gpsro": {"variables": ["gpsro_*"]},
    }
    for name, values in overrides.items():
        groups[name].update(values)
    return groups


def _structured(data_indices, groups=None, **config) -> StructuredObsDropout:
    full = {"multi_step": 2, "groups": groups or _groups(), **config}
    return StructuredObsDropout(config=DictConfig(full), data_indices=data_indices)


def _batch(batch: int = 4, time: int = 4, grid: int = 200) -> torch.Tensor:
    x = torch.ones(batch, time, 1, grid, len(NAMES))
    lat = torch.linspace(-math.pi / 2, math.pi / 2, grid)
    lon = torch.linspace(0, 2 * math.pi, grid)
    for idx, values in zip(FORCING[:4], (lat.cos(), lat.sin(), lon.cos(), lon.sin()), strict=True):
        x[..., idx] = values
    return x


def _transform(dropout, x: torch.Tensor, name_to_index=NAME_TO_INDEX, **kwargs) -> torch.Tensor:
    return dropout.transform(x.clone(), name_to_index=name_to_index, **kwargs)


# ── StructuredObsDropout ───────────────────────────────────────────────


def test_stream_drops_whole_group_including_corrector(data_indices) -> None:
    dropout = _structured(data_indices, _groups(mw_temp={"stream_prob": 1.0}))
    out = _transform(dropout, _batch())
    assert torch.isnan(out[:, :2, ..., MWT]).all()
    assert not torch.isnan(out[:, 2:]).any()  # beyond multi_step: untouched
    assert not torch.isnan(out[..., SONDE + GPSRO + FORCING]).any()


def test_multi_step_none_covers_every_step(data_indices) -> None:
    dropout = StructuredObsDropout(
        config=DictConfig({"groups": _groups(mw_temp={"stream_prob": 1.0})}),
        data_indices=data_indices,
    )
    assert torch.isnan(_transform(dropout, _batch())[..., MWT]).all()


def test_cell_mask_shared_across_variables_and_time(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _structured(data_indices, _groups(radiosonde={"cell_prob": 0.4}))
    out = _transform(dropout, _batch(time=6))
    nan = torch.isnan(out[:, :2, ..., SONDE])
    reference = nan[:, :1, ..., :1]
    assert (nan == reference).all()  # same cells for every sonde variable and input step
    assert reference.float().mean().item() == pytest.approx(0.4, abs=0.06)


def test_block_dropout_respects_radius(data_indices) -> None:
    torch.manual_seed(0)
    radius = 2000.0
    dropout = _structured(data_indices, _groups(gpsro={"n_blocks": 1, "block_radius_km": radius}))
    x = _batch(batch=8, grid=400)
    out = _transform(dropout, x)
    coordinates = {key: NAME_TO_INDEX[name] for key, name in dropout.coordinate_variables.items()}
    xyz = dropout._unit_vectors(x, coordinates)
    for b in range(x.shape[0]):
        dropped = torch.isnan(out[b, 0, 0, :, GPSRO[0]])
        assert dropped.any()
        # every dropped cell lies within 2 * radius of every other (one disc)
        idx = dropped.nonzero().squeeze(-1)
        cos_angle = xyz[b, idx] @ xyz[b, idx].T
        assert (cos_angle >= math.cos(2 * radius / 6371.0) - 1e-5).all()


def test_max_streams_dropped(data_indices) -> None:
    torch.manual_seed(0)
    groups = _groups(radiosonde={"stream_prob": 1.0}, mw_temp={"stream_prob": 1.0}, gpsro={"stream_prob": 1.0})
    dropout = _structured(data_indices, groups, max_streams_dropped=2)
    out = _transform(dropout, _batch(batch=16))
    per_group = [torch.isnan(out[:, 0, 0, 0, idx[0]]) for idx in (SONDE, MWT, GPSRO)]
    assert (torch.stack(per_group, dim=1).sum(dim=1) == 2).all()


def test_eval_and_target_views_are_untouched(data_indices) -> None:
    dropout = _structured(data_indices, _groups(radiosonde={"stream_prob": 1.0, "cell_prob": 1.0}))
    # Targets keep their missing values for the loss and are pre-processed with skip_imputation.
    assert not torch.isnan(_transform(dropout, _batch(), skip_imputation=True)).any()
    # Observations that keep their missing values but feed the model still get dropout.
    assert torch.isnan(_transform(dropout, _batch(), skip_imputation=True, apply_dropout=True)[:, :2, ..., SONDE]).all()
    dropout.eval()
    assert not torch.isnan(_transform(dropout, _batch())).any()


def test_view_with_a_subset_of_the_variables(data_indices) -> None:
    """Target forcings carry only some variables, at their own positions."""
    dropout = _structured(data_indices, _groups(mw_temp={"stream_prob": 1.0}, radiosonde={"stream_prob": 1.0}))
    names = ["lsm", "mwt_2", "t_850"]
    x = torch.ones(2, 1, 1, 10, len(names))
    out = _transform(dropout, x, name_to_index={n: i for i, n in enumerate(names)})
    assert torch.isnan(out[..., 1:]).all()
    assert not torch.isnan(out[..., 0]).any()


def test_tabular_sample(data_indices) -> None:
    """A tabular view is processed one (ensemble, rows, variables) sample at a time."""
    torch.manual_seed(0)
    dropout = _structured(data_indices, _groups(radiosonde={"cell_prob": 0.5}))
    sample = torch.ones(1, 300, len(NAMES))
    out = _transform(dropout, sample)
    rows = torch.isnan(out[0, :, SONDE])
    assert (rows == rows[:, :1]).all()  # a dropped row loses every sonde variable
    assert rows[:, 0].float().mean().item() == pytest.approx(0.5, abs=0.1)
    assert not torch.isnan(out[..., MWT + GPSRO + FORCING]).any()


def test_through_processors_on_a_source(data_indices) -> None:
    dropout = _structured(data_indices, _groups(mw_temp={"stream_prob": 1.0}))
    processors = Processors([["obs_dropout", dropout]])
    source = GriddedSource(
        name="data",
        variables=NAMES,
        layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
        statistics={},
        data=_batch(),
        coordinates=torch.zeros(200, 2),
    )
    out = processors(source, in_place=False)
    assert torch.isnan(out.data[:, :2, ..., MWT]).all()
    assert not torch.isnan(source.data).any()  # in_place=False leaves the input alone
    assert not torch.isnan(processors(source, in_place=False, skip_imputation=True).data).any()


def test_dropout_prob_scales_probabilities(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _structured(data_indices, _groups(radiosonde={"cell_prob": 0.8}))
    dropout.dropout_prob = 0.25  # as set by DropoutScheduler
    out = _transform(dropout, _batch(batch=8, grid=500))
    assert torch.isnan(out[:, 0, 0, :, SONDE[0]]).float().mean().item() == pytest.approx(0.2, abs=0.04)
    dropout.dropout_prob = 0.0
    assert not torch.isnan(_transform(dropout, _batch())).any()
    assert len(dropout.dropout_indices) == len(SONDE + MWT + GPSRO)


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ({"a": {"variables": ["t_*"]}, "b": {"variables": ["t_500"]}}, "disjoint"),
        ({"a": {"variables": ["nothing_*"]}}, "matches no input variable"),
        ({"a": {"variables": ["*_latitude"]}}, "forcing"),
        ({"a": {"variables": ["t_*"], "cell_prob": 1.5}}, r"\[0, 1\]"),
        ({"a": {"variables": ["t_*"], "n_blocks": 1}}, "block_radius_km"),
    ],
)
def test_invalid_config_raises(data_indices, groups, match) -> None:
    with pytest.raises(ValueError, match=match):
        _structured(data_indices, groups)


def test_schemas_accept_example_configs() -> None:
    StructuredObsDropoutSchema.model_validate(
        {
            "max_streams_dropped": 2,
            "groups": {"radiosonde": {"variables": ["z_*"], "cell_prob": 0.3, "n_blocks": 2, "block_radius_km": 1500}},
        },
    )
    RandomSpatialDropoutSchema.model_validate({"dropout_prob": 0.15, "dropout_variables": ["z_500"]})


# ── RandomSpatialDropout ───────────────────────────────────────────────


def _spatial(data_indices, **config) -> RandomSpatialDropout:
    return RandomSpatialDropout(config=DictConfig({"multi_step": 2, **config}), data_indices=data_indices)


def test_spatial_dropout_fraction_and_input_steps(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _spatial(data_indices, dropout_prob=0.3)
    out = _transform(dropout, _batch(batch=8, grid=500))
    non_forcing = SONDE + MWT + GPSRO
    assert torch.isnan(out[:, :2, ..., non_forcing]).float().mean().item() == pytest.approx(0.3, abs=0.03)
    assert not torch.isnan(out[:, 2:]).any()
    assert not torch.isnan(out[..., FORCING]).any()  # forcings are protected by default


def test_spatial_dropout_explicit_variables(data_indices) -> None:
    dropout = _spatial(data_indices, dropout_prob=1.0, dropout_variables=["t_500", "lsm", "missing"])
    out = _transform(dropout, _batch())
    assert torch.isnan(out[:, :2, ..., [1, 11]]).all()
    assert not torch.isnan(out[..., [0, 2, 3]]).any()


def test_spatial_dropout_noop_cases(data_indices) -> None:
    assert not torch.isnan(_transform(_spatial(data_indices, dropout_prob=0.0), _batch())).any()
    dropout = _spatial(data_indices, dropout_prob=1.0)
    assert not torch.isnan(_transform(dropout, _batch(), skip_imputation=True)).any()
    dropout.eval()
    assert not torch.isnan(_transform(dropout, _batch())).any()
    x = _batch()
    assert dropout.inverse_transform(x) is x


def test_spatial_dropout_invalid_prob(data_indices) -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _spatial(data_indices, dropout_prob=1.5)

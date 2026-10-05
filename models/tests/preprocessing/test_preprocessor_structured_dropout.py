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

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.structured_dropout import StructuredObsDropout
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
    return IndexCollection(data_config=config, name_to_index={n: i for i, n in enumerate(NAMES)})


def _groups(**overrides) -> dict:
    groups = {
        "radiosonde": {"variables": ["z_*", "t_*"]},
        "mw_temp": {"variables": ["mwt_*"]},
        "gpsro": {"variables": ["gpsro_*"]},
    }
    for name, values in overrides.items():
        groups[name].update(values)
    return groups


def _make(data_indices, groups=None, **config) -> StructuredObsDropout:
    full = {"multi_step": 2, "groups": groups or _groups(), **config}
    return StructuredObsDropout(config=DictConfig(full), data_indices=data_indices)


def _batch(batch: int = 4, time: int = 4, grid: int = 200, ensemble: bool = True) -> torch.Tensor:
    shape = (batch, time, 1, grid, len(NAMES)) if ensemble else (batch, time, grid, len(NAMES))
    x = torch.ones(shape)
    lat = torch.linspace(-math.pi / 2, math.pi / 2, grid)
    lon = torch.linspace(0, 2 * math.pi, grid)
    for idx, values in zip(FORCING[:4], (lat.cos(), lat.sin(), lon.cos(), lon.sin()), strict=True):
        x[..., idx] = values
    return x


def test_stream_drops_whole_group_including_corrector(data_indices) -> None:
    dropout = _make(data_indices, _groups(mw_temp={"stream_prob": 1.0}))
    out = dropout.transform(_batch(), in_place=False)
    assert torch.isnan(out[:, :2, ..., MWT]).all()
    assert not torch.isnan(out[:, 2:]).any()  # target steps untouched
    assert not torch.isnan(out[..., SONDE + GPSRO + FORCING]).any()


def test_cell_mask_shared_across_variables_and_time(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make(data_indices, _groups(radiosonde={"cell_prob": 0.4}))
    out = dropout.transform(_batch(time=6), in_place=False)
    nan = torch.isnan(out[:, :2, ..., SONDE])
    reference = nan[:, :1, ..., :1]
    assert (nan == reference).all()  # same cells for every sonde variable and input step
    assert reference.float().mean().item() == pytest.approx(0.4, abs=0.06)


def test_block_dropout_respects_radius(data_indices) -> None:
    torch.manual_seed(0)
    radius = 2000.0
    dropout = _make(data_indices, _groups(gpsro={"n_blocks": 1, "block_radius_km": radius}))
    x = _batch(batch=8, grid=400)
    out = dropout.transform(x.clone(), in_place=False)
    xyz = dropout._unit_vectors(x)
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
    dropout = _make(data_indices, groups, max_streams_dropped=2)
    out = dropout.transform(_batch(batch=16), in_place=False)
    per_group = [torch.isnan(out[:, 0, 0, 0, idx[0]]) for idx in (SONDE, MWT, GPSRO)]
    assert (torch.stack(per_group, dim=1).sum(dim=1) == 2).all()


def test_only_valid_values_and_four_dim_input(data_indices) -> None:
    dropout = _make(data_indices, _groups(radiosonde={"stream_prob": 1.0}))
    x = _batch(ensemble=False)
    x[:, 0, :10, SONDE[0]] = torch.nan
    out = dropout.transform(x, in_place=False)
    assert torch.isnan(out[:, :2, :, SONDE]).all()
    assert not torch.isnan(out[:, :2, :, MWT + GPSRO + FORCING]).any()


def test_eval_noop(data_indices) -> None:
    dropout = _make(data_indices, _groups(radiosonde={"stream_prob": 1.0, "cell_prob": 1.0}))
    dropout.eval()
    assert not torch.isnan(dropout.transform(_batch(), in_place=False)).any()


def test_dropout_prob_scales_probabilities(data_indices) -> None:
    torch.manual_seed(0)
    dropout = _make(data_indices, _groups(radiosonde={"cell_prob": 0.8}))
    dropout.dropout_prob = 0.25  # as set by DropoutScheduler
    out = dropout.transform(_batch(batch=8, grid=500), in_place=False)
    assert torch.isnan(out[:, 0, 0, :, SONDE[0]]).float().mean().item() == pytest.approx(0.2, abs=0.04)
    dropout.dropout_prob = 0.0
    assert not torch.isnan(dropout.transform(_batch(), in_place=False)).any()
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
        _make(data_indices, groups)


def test_schema_accepts_example_config() -> None:
    StructuredObsDropoutSchema.model_validate(
        {
            "multi_step": 6,
            "max_streams_dropped": 2,
            "groups": {"radiosonde": {"variables": ["z_*"], "cell_prob": 0.3, "n_blocks": 2, "block_radius_km": 1500}},
        },
    )

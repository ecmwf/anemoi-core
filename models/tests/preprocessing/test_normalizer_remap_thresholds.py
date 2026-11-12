# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Normalised thresholds follow the normaliser remap as in earlier releases (main @ 9b0cf9b4e).

There the normaliser rewrote the shared statistics in place when it was built, so a
NormalizedReluPostprocessor listed after it, and every NormalizedReluBounding (built with the model,
after the processors), normalised the thresholds of remapped variables with the remapped statistics;
a post-processor listed before the normaliser kept the variable's own statistics. The expected values
below were produced by running main's code on this setup.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.layers.bounding import NormalizedLeakyReluBounding
from anemoi.models.layers.bounding import NormalizedReluBounding
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalization import remapped_statistics
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.preprocessing.postprocessor import NormalizedReluPostprocessor

NAMES = ["tp", "cp", "t"]
STATISTICS = {
    "mean": np.array([2.0, 1.0, 280.0]),
    "stdev": np.array([4.0, 0.5, 10.0]),
    "minimum": np.array([0.0, 0.0, 200.0]),
    "maximum": np.array([50.0, 20.0, 320.0]),
}
# A physical threshold of 0.5 on cp, mean-std normalised.
OWN = (0.5 - 1.0) / 0.5  # cp's statistics: -1.0
REMAPPED = (0.5 - 2.0) / 4.0  # tp's statistics, through the remap cp: tp: -0.375


def _data_indices() -> IndexCollection:
    return IndexCollection(
        DictConfig({"forcing": [], "diagnostic": ["tp", "cp"], "target": []}), {n: i for i, n in enumerate(NAMES)}
    )


def _view(payload: torch.Tensor) -> GriddedSource:
    return GriddedSource(
        name="data",
        data=payload.reshape(1, 1, 1, len(NAMES)).clone(),
        variables=list(NAMES),
        statistics={key: value.copy() for key, value in STATISTICS.items()},
        coordinates=torch.zeros(1, 2),
        layout=TensorLayout(batch=0, time=1, grid=2, variables=3),
    )


def _processors(normalizer_first: bool) -> tuple[Processors, Processors]:
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std", "remap": {"cp": "tp"}}))
    relu = NormalizedReluPostprocessor(
        config=DictConfig({"default": "none", 0.5: ["cp"], "normalizer": "mean-std"}), data_indices=_data_indices()
    )
    chain = [["normalizer", normalizer], ["relu", relu]]
    if not normalizer_first:
        chain = chain[::-1]
    return Processors(chain), Processors(chain, inverse=True)


@pytest.mark.parametrize(("normalizer_first", "threshold"), [(True, REMAPPED), (False, OWN)])
def test_postprocessor_threshold_follows_remap_of_earlier_normalizer(normalizer_first: bool, threshold: float) -> None:
    _, post = _processors(normalizer_first)
    relu = post.processors["relu"]
    # The ReLU acts on the normalised cp: anything below the normalised threshold is lifted to it.
    x = torch.full((len(NAMES),), -100.0)
    out = relu.inverse_transform(x.clone(), statistics=STATISTICS, name_to_index={n: i for i, n in enumerate(NAMES)})
    assert out[1].item() == threshold  # bit-identical to main
    assert torch.equal(out[[0, 2]], x[[0, 2]])


@pytest.mark.parametrize("cls", [NormalizedReluBounding, NormalizedLeakyReluBounding])
def test_bounding_threshold_follows_dataset_remap(cls) -> None:
    pre, _ = _processors(normalizer_first=False)  # the boundings see the remap whatever the order
    bounding = cls(variables=["cp"], min_val=[0.5], normalizer=["mean-std"])
    model = torch.nn.Module()
    model.boundings = torch.nn.ModuleDict({"data": torch.nn.ModuleList([bounding])})

    AnemoiModelInterface._share_statistics_remap(model, torch.nn.ModuleDict({"data": pre}))

    view = _view(torch.zeros(len(NAMES)))
    norm_min_val = bounding._compute_norm_min_val(view.name_to_index, view.statistics)
    assert norm_min_val.tolist() == [REMAPPED]
    # Without the remap (e.g. a bounding not built by the interface): cp's own statistics.
    norm_min_val = cls(variables=["cp"], min_val=[0.5], normalizer=["mean-std"])._compute_norm_min_val(
        view.name_to_index, view.statistics
    )
    assert norm_min_val.tolist() == [OWN]


def test_remapped_statistics_copies_and_uses_original_sources() -> None:
    name_to_index = {n: i for i, n in enumerate(NAMES)}
    statistics = {key: value.copy() for key, value in STATISTICS.items()}

    # A chained remap takes the original statistics of each source, as the normaliser does.
    remapped = remapped_statistics(statistics, name_to_index, {"cp": "tp", "tp": "t"}, "test")

    assert remapped["mean"].tolist() == [280.0, 2.0, 280.0]
    assert all(np.array_equal(statistics[key], STATISTICS[key]) for key in STATISTICS)  # unchanged
    with pytest.raises(ValueError, match="'tp' is not in this view"):
        remapped_statistics(statistics, {"cp": 0}, {"cp": "tp"}, "test")

# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.preprocessing.normalizer import InputNormalizer


def _batch(variables):
    return Batch(
        data={"grid": torch.zeros(2, 1, 1, 3, 2)},
        layouts={"grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables=variables,
    )


@pytest.mark.parametrize("variables", [{}, {"grid": ["a"]}, {"grid": ["a", "a"]}])
def test_view_requires_matching_unique_variable_names(variables):
    with pytest.raises(ValueError, match="variable names|variable channels|unique variable names"):
        _batch(variables)["grid"]


def test_statistics_and_coordinates_are_required_by_the_consuming_operation():
    view = _batch({"grid": ["a", "b"]})["grid"]
    assert view.statistics == {}
    assert view.coordinates is None
    # Metadata-free arithmetic is legitimate; graph construction needs geometry.
    torch.testing.assert_close(view.apply_func(lambda data, **kwargs: data + 1).data, torch.ones_like(view.data))
    with pytest.raises(ValueError, match="requires coordinates"):
        view.flatten()
    normalizer = InputNormalizer({"default": "mean-std"})
    with pytest.raises(ValueError, match="requires statistics"):
        normalizer.get_norm_parameters(view.statistics, view.name_to_index, torch.device("cpu"))


def test_normalizer_requires_only_statistics_used_by_its_method():
    normalizer = InputNormalizer({"default": "std"})
    multiplier, offset = normalizer.get_norm_parameters(
        {"stdev": torch.tensor([2.0, 4.0])}, {"a": 0, "b": 1}, torch.device("cpu")
    )
    torch.testing.assert_close(multiplier, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(offset, torch.zeros(2))
    identity = InputNormalizer({"default": "none"})
    multiplier, offset = identity.get_norm_parameters({}, {"a": 0, "b": 1}, torch.device("cpu"))
    torch.testing.assert_close(multiplier, torch.ones(2))
    torch.testing.assert_close(offset, torch.zeros(2))


def test_update_source_preserves_layout_and_coordinate_staticness():
    batch = _batch({"grid": ["a", "b"]})
    view = batch["grid"].clone(coordinates=torch.zeros(3, 2), coordinates_are_static=True)
    fixed = batch.update_source("grid", view)
    assert fixed.is_static_coords("grid")
    assert fixed["grid"].coordinates_are_static
    assert fixed["grid"].flatten().batch_sizes is None

    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=-2, variables=-1)
    moving = view.clone(layout=layout, coordinates=torch.zeros(2, 3, 2), coordinates_are_static=False)
    updated = fixed.update_source("grid", moving)
    assert not updated.is_static_coords("grid")
    assert not updated["grid"].coordinates_are_static
    assert updated["grid"].layout == layout
    assert updated["grid"].flatten().batch_sizes == (3, 3)
    assert fixed.is_static_coords("grid")


def test_model_output_cast_and_variable_metadata_agree():
    from types import SimpleNamespace

    from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

    model = AnemoiModelEncProcDec.__new__(AnemoiModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.data_indices = {
        "grid": SimpleNamespace(
            name_to_index={"a": 0, "b": 1}, model=SimpleNamespace(output=SimpleNamespace(ordered_names=["b"]))
        )
    }
    model.boundings = {"grid": torch.nn.Identity()}
    target = _batch({"grid": ["a", "b"]})["grid"].clone(statistics={"mean": torch.tensor([1.0, 2.0])})
    model.statistics = {"grid": target.statistics}
    output = model._assemble_output(torch.ones(6, 1, dtype=torch.bfloat16), None, target, torch.bfloat16, "grid")
    assert output.dtype == torch.float32
    assert output.variables == ["b"]
    torch.testing.assert_close(output.statistics["mean"], torch.tensor([2.0]))
    torch.testing.assert_close(output.data, torch.ones(2, 1, 1, 3, 1))


@pytest.mark.parametrize("shape", [(6,), (1, 2, 3, 2), (3, 3), (1, 3, 2)])
def test_gridded_flatten_rejects_invalid_coordinate_rank_or_shape(shape):
    view = _batch({"grid": ["a", "b"]})["grid"].clone(coordinates=torch.zeros(shape))
    with pytest.raises(ValueError, match="coordinates must have shape"):
        view.flatten()

# (C) Copyright 2026 Anemoi contributors.

import torch
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.variable.variabletokenizer import MeanPoolingTransform
from anemoi.models.variable.variablevocabular import VariableVocabulary


def _indices(*names: str) -> IndexCollection:
    return IndexCollection(
        OmegaConf.create({"forcing": [], "diagnostic": [], "target": []}),
        {name: index for index, name in enumerate(names)},
    )


def _metadata(param: str, *, units: str, levtype: str = "sfc") -> dict:
    return {
        "mars": {"param": param, "levtype": levtype, "time": 0, "step": 0},
        "units": units,
    }


def test_pressure_and_precipitation_semantics_follow_issue_1411() -> None:
    metadata = {
        "msl": _metadata("msl", units="Pa"),
        "mslp": _metadata("mslp", units="Pa"),
        "sp": _metadata("sp", units="Pa"),
        "tp": _metadata("tp", units="m"),
    }
    vocabulary = VariableVocabulary.from_foundation(_indices(*metadata), metadata)

    for name in ("msl", "mslp"):
        specification = vocabulary.specification_by_name[name]
        assert specification.param == "p"
        assert specification.vertical_coordinate.type == "height_above_sea"
        assert specification.vertical_coordinate.level == 0.0
        assert specification.value_unit == "Pa"

    surface_pressure = vocabulary.specification_by_name["sp"]
    assert surface_pressure.param == "p"
    assert surface_pressure.vertical_coordinate.type == "height_above_ground"
    assert surface_pressure.vertical_coordinate.level == 0.0

    precipitation = vocabulary.specification_by_name["tp"]
    assert precipitation.param == "tp"
    assert precipitation.vertical_coordinate.type == "height_above_ground"
    assert precipitation.vertical_coordinate.level == 0.0


def test_vocabulary_is_union_of_domain_fields() -> None:
    data_indices = {"domain_a": _indices("2t"), "domain_b": _indices("tp")}
    metadata = {
        "domain_a": {"2t": _metadata("2t", units="K")},
        "domain_b": {"tp": _metadata("tp", units="m")},
    }

    vocabulary = VariableVocabulary.from_datasets(data_indices, metadata)

    assert set(vocabulary.specification_by_name) == {"2t", "tp"}
    assert vocabulary.specification_by_name["2t"].value_unit == "K"


def test_vocabulary_serialization_preserves_value_units() -> None:
    metadata = {"2t": _metadata("2t", units="K")}
    vocabulary = VariableVocabulary.from_foundation(_indices("2t"), metadata)

    restored = VariableVocabulary.from_dict(vocabulary.to_dict())

    assert restored.specification_by_name["2t"].value_unit == "K"


def test_tokenizer_has_fixed_width_for_different_domain_field_sets() -> None:
    data_indices = {"domain_a": _indices("2t", "sp"), "domain_b": _indices("tp")}
    metadata = {
        "domain_a": {
            "2t": _metadata("2t", units="K"),
            "sp": _metadata("sp", units="Pa"),
        },
        "domain_b": {"tp": _metadata("tp", units="m")},
    }
    vocabulary = VariableVocabulary.from_datasets(data_indices, metadata)
    tokenizer = MeanPoolingTransform(emb_dim=8, out_dim=4, vocabulary=vocabulary)

    domain_a = tokenizer(torch.ones(2, 3, 1, 5, 2), ["2t", "sp"])
    domain_b = tokenizer(torch.ones(2, 3, 1, 5, 1), ["tp"])

    assert domain_a.shape == domain_b.shape == (10, 12)

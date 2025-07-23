# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.schemas.data_processor import NormalizerSchema
from anemoi.models.schemas.data_processor import PreprocessorSchema


def test_preprocessor_with_raw_dict():
    raw_config = {"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"], "mean-std": ["q"]}
    schema = PreprocessorSchema(_target_="anemoi.models.preprocessing.normalizer.InputNormalizer", config=raw_config)

    assert schema.target_ == "anemoi.models.preprocessing.normalizer.InputNormalizer"
    assert schema.config == raw_config


def test_preprocessor_with_normalizer_instance():
    normalizer_instance = NormalizerSchema(default="std", remap={"c": "d"})
    schema = PreprocessorSchema(
        _target_="anemoi.models.preprocessing.normalizer.InputNormalizer", config=normalizer_instance
    )

    assert schema.target_ == "anemoi.models.preprocessing.normalizer.InputNormalizer"
    assert isinstance(schema.config, NormalizerSchema)
    assert schema.config.default == "std"
    assert schema.config.remap == {"c": "d"}

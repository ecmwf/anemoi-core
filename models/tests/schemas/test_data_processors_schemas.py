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

# ✅ Test with raw dict
raw_input = PreprocessorSchema(
    _target_="anemoi.models.preprocessing.normalizer.InputNormalizer",
    config={"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"], "mean-std": ["q"]},
)
print("Parsed from dict:", raw_input)

# ✅ Test with NormalizerSchema instance
normalizer_instance = NormalizerSchema(default="std", remap={"c": "d"})
model_input = PreprocessorSchema(
    _target_="anemoi.models.preprocessing.normalizer.InputNormalizer", config=normalizer_instance
)
print("Parsed from model instance:", model_input)

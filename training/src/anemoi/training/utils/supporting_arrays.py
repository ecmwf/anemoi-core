# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from hydra.utils import instantiate


def build_combined_supporting_arrays(config, graph_data: dict, supporting_arrays: dict) -> dict:
    """Merge output-mask supporting arrays into supporting_arrays."""
    combined = supporting_arrays.copy()
    for name, data in graph_data.items():
        mask = instantiate(config.model.output_mask, graph_data=data)
        combined[name].update(mask.supporting_arrays)
    return combined

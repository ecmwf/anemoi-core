# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from hydra.utils import instantiate
from torch_geometric.data import HeteroData


def build_combined_supporting_arrays(
    config: Any,
    graph_data: HeteroData,
    supporting_arrays: dict,
    dataset_names: list[str] | None = None,
) -> dict:
    """Merge output-mask supporting arrays into supporting_arrays."""
    combined = {name: arrays.copy() for name, arrays in supporting_arrays.items()}
    if dataset_names is None:
        from anemoi.models.utils.config import get_multiple_datasets_config

        dataset_names = list(get_multiple_datasets_config(config.data).keys())
    backbone_cfg = getattr(getattr(config, "model", None), "backbone", None)
    output_mask_cfg = getattr(backbone_cfg, "output_mask", None) if backbone_cfg is not None else None
    for name in dataset_names:
        combined.setdefault(name, {})
        if output_mask_cfg is not None:
            mask = instantiate(output_mask_cfg, nodes=graph_data[name])
            combined[name].update(mask.supporting_arrays)
    return combined

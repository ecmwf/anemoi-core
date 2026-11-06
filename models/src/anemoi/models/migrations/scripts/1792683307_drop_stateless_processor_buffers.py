# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE

# The persistent buffers ``InputNormalizer`` used to register, under any pre-/post-processor
# (including the tendency processors), whatever the processor is called in the config.
_PROCESSOR_BUFFER = re.compile(
    r"^model\.(pre|post)_processors(_tendencies)?\..*\.(_norm_mul|_norm_add|_input_idx|_output_idx|_model_output_idx)$"
)
# The sin/cos node coordinates ``NodeTrainableParameters`` used to register, one per node set.
_NODE_LATLONS = re.compile(r"^model\.model\.node_attributes\.latlons_[^.]+$")


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    Drops buffers that are now computed at run time and no longer stored in the model.

    - The pre- and post-processors are stateless: the normaliser takes its statistics and
      variable indices from the data. This removes the ``_norm_mul``, ``_norm_add``,
      ``_input_idx``, ``_output_idx`` and ``_model_output_idx`` buffers of every processor.
    - ``NodeTrainableParameters`` no longer keeps the node coordinates: they come from the
      graph. This removes ``model.model.node_attributes.latlons_<nodes>``; the trainable
      node tensors next to them are kept.

    Before: ``model.pre_processors.data.processors.normalizer._norm_mul`` (removed)
            ``model.model.node_attributes.latlons_data``                (removed)

    Nothing is lost: every removed buffer is derived from the statistics, the data indices or
    the graph. Checkpoints without these keys are returned unchanged.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    state_dict = ckpt["state_dict"]
    for key in [k for k in state_dict if _PROCESSOR_BUFFER.match(k) or _NODE_LATLONS.match(k)]:
        del state_dict[key]
    return ckpt

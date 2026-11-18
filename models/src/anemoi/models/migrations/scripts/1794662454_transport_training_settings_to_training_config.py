# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import OmegaConf
from omegaconf import open_dict

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

# Training-only settings that sat in the model's transport config.
_TRAINING_NOISE_RANGE = ("sigma_max", "sigma_min", "rho")


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    Moves the transport training settings into the training config of the stored run config:

    - ``model.model.transport.training_condition`` moves to ``training.transport.training_condition``, with
      its values resolved, since they usually point at the removed ``model.model.transport.sigma_*`` and ``rho``;
    - ``model.model.transport.sigma_max``, ``sigma_min`` and ``rho`` are removed;
    - ``training.transport.objective`` and ``training.transport.prediction_mode`` are removed: the objective is
      read from the model config and the prediction mode from the model class.

    Checkpoints of other models are returned unchanged.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    config = ckpt.get("hyper_parameters", {}).get("config")
    model_transport = OmegaConf.select(config, "model.model.transport") if config is not None else None
    if model_transport is None:
        return ckpt

    with open_dict(config):
        training_condition = model_transport.get("training_condition")
        if training_condition is not None:
            training_condition = OmegaConf.to_container(training_condition, resolve=True)
        for key in ("training_condition", *_TRAINING_NOISE_RANGE):
            model_transport.pop(key, None)

        training_transport = OmegaConf.select(config, "training.transport")
        if training_transport is None:
            config.training.transport = {}
            training_transport = config.training.transport
        training_transport.pop("objective", None)
        training_transport.pop("prediction_mode", None)
        if training_condition is not None and "training_condition" not in training_transport:
            training_transport.training_condition = training_condition
    return ckpt

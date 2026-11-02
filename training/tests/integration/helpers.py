# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Helper functions shared by the integration tests."""

from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.utils.testing import GetTestData

TASK_SPECIFIC_TIMESTEP_KEYS = {
    "offset-forecaster": {"input_offsets", "output_offsets", "rollout_shift", "advance_map"},
}


def assert_keys_exist(data: dict, schema: dict, path: str = "root", skip_keys: set[str] | None = None) -> None:
    """Recursively check that the metadata dictionary conforms to the expected schema.

    This is a simplified schema validation that only checks for the presence of expected keys.
    Note that this does not ensure that changes in anemoi-core do not break anemoi-inference.
    """
    if skip_keys is None:
        task = data.get("metadata_inference", []).get("task")
        assert task is not None, "metadata_inference.task must be populated"
        all_task_keys = set().union(*TASK_SPECIFIC_TIMESTEP_KEYS.values())
        skip_keys = all_task_keys - TASK_SPECIFIC_TIMESTEP_KEYS.get(task, set())

    for key, subschema in schema.items():
        if key == "__datasets__":
            dataset_names = data.get("dataset_names", [])
            for ds in dataset_names:
                assert ds in data, f"{path}: dataset '{ds}' missing"
                assert_keys_exist(data[ds], subschema, f"{path}.{ds}", skip_keys)
            continue

        if key in skip_keys:
            continue

        assert key in data, f"{path}: missing key '{key}'"

        if isinstance(subschema, dict):
            assert isinstance(data[key], dict), f"{path}.{key} should be dict"
            assert_keys_exist(data[key], subschema, f"{path}.{key}", skip_keys)

        if subschema is list:
            assert isinstance(data[key], list), f"{path}.{key} should be list"


def _get_multiscale_cfgs(training_loss_cfg: DictConfig) -> list[DictConfig]:
    """Extract multiscale_config dicts that contain loss_matrices."""
    multiscale_cfg = training_loss_cfg.get("multiscale_config")
    if multiscale_cfg is not None and "loss_matrices" in multiscale_cfg:
        return [multiscale_cfg]
    if "losses" in training_loss_cfg:
        results = []
        for sub_loss in training_loss_cfg.losses:
            mc = sub_loss.get("multiscale_config")
            if mc is not None and "loss_matrices" in mc:
                results.append(mc)
        return results
    return []


def handle_truncation_matrices(cfg: DictConfig, get_test_data: GetTestData) -> DictConfig:
    url_loss_matrices = cfg.system.input.loss_matrices_path
    tmp_path_loss_matrices = None

    training_losses_cfg = get_multiple_datasets_config(cfg.training.training_loss)
    for dataset_name, training_loss_cfg in training_losses_cfg.items():
        multiscale_cfgs = _get_multiscale_cfgs(training_loss_cfg)

        for multiscale_cfg in multiscale_cfgs:
            for file in multiscale_cfg.get("loss_matrices") or []:
                if file is not None:
                    tmp_path_loss_matrices = get_test_data(url_loss_matrices + file)
            if tmp_path_loss_matrices is not None:
                OmegaConf.set_struct(multiscale_cfg, False)
                multiscale_cfg.loss_matrices_path = str(Path(tmp_path_loss_matrices).parent)

        if tmp_path_loss_matrices is not None:
            resolved_path = str(Path(tmp_path_loss_matrices).parent)
            cfg.system.input.loss_matrices_path = Path(tmp_path_loss_matrices).parent

            val_multiscale_cfg = cfg.training.validation_metrics.datasets[dataset_name].multiscale.multiscale_config
            OmegaConf.set_struct(val_multiscale_cfg, False)
            val_multiscale_cfg.loss_matrices_path = resolved_path
        cfg.training.training_loss.datasets[dataset_name] = training_loss_cfg
    return cfg

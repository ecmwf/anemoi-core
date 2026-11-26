# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from omegaconf import DictConfig

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


@skip_if_offline
@pytest.mark.metadata
def test_metadata_reading(gnn_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    # Imported here so that collecting the tests does not require anemoi-metadata
    from anemoi.metadata import Metadata

    cfg, url = gnn_config
    get_test_archive(url)

    AnemoiTrainer(cfg).train()
    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)

    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert len(run_dirs) == 1, (
        f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"
    )

    checkpoint_dir = run_dirs[0]
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, f"No checkpoint files found in directory: {checkpoint_dir}"

    # The inference checkpoint is written next to the lightning one with an "inference-" prefix
    inference = [f for f in checkpoint_files if f.name.startswith("inference-")]
    lightning = [f for f in checkpoint_files if not f.name.startswith("inference-")]
    to_check = inference[:1] + lightning[:1]
    assert to_check, f"No checkpoint to check in {checkpoint_dir}"

    for checkpoint in to_check:
        metadata = Metadata.from_checkpoint(checkpoint)

        # Training writes format "3.0" without component_versions; the package infers 1.0 for each component
        assert metadata.version == "3.0", checkpoint.name
        assert metadata.component_versions == {"data": "1.0", "task": "1.0", "lineage": "1.0"}, checkpoint.name
        assert metadata.dataset_names == ["data"], checkpoint.name

        task = metadata.task
        assert task.task_name == "forecaster", checkpoint.name
        assert task.timestep() == "6h", checkpoint.name
        assert task.n_step_input == cfg.task.multistep_input, checkpoint.name
        assert task.n_step_output == cfg.task.multistep_output, checkpoint.name
        assert task.input_relative_date_indices() == list(range(task.n_step_input)), checkpoint.name
        assert task.output_relative_date_indices() == [
            task.n_step_input - 1 + i for i in range(1, 1 + task.n_step_output)
        ]

        assert metadata.lineage.uuid, checkpoint.name
        assert metadata.lineage.parent_uuid is None, checkpoint.name
        assert metadata.lineage.entries == [], checkpoint.name

        dataset = metadata.dataset("data")
        data = dataset.data
        assert data.data_type() == "gridded", checkpoint.name
        # Grid of the pinned test archive dataset
        assert data.grid_size() == 40320, checkpoint.name

        latitudes, longitudes = dataset.arrays.latitudes, dataset.arrays.longitudes
        assert latitudes is not None and longitudes is not None, checkpoint.name
        assert data.grid_size() == len(latitudes), checkpoint.name
        assert latitudes.shape == longitudes.shape == (data.grid_size(),), checkpoint.name

        variables = data.variables()
        assert variables, checkpoint.name
        assert len(variables) == len(set(data.variable_indices()) | set(data.output_variable_indices())), (
            checkpoint.name
        )
        assert sorted(data.variable_indices().values()) == list(range(len(data.variable_indices()))), checkpoint.name
        assert data.prognostic_input_mask(), checkpoint.name

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
    cfg, url = gnn_config
    get_test_archive(url)

    AnemoiTrainer(cfg).train()
    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)

    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert (
        len(run_dirs) == 1
    ), f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"

    checkpoint_dir = run_dirs[0]
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, f"No checkpoint files found in directory: {checkpoint_dir}"

    from anemoi.metadata import Metadata

    metadata = Metadata.from_checkpoint(checkpoint_files[0])
    assert metadata is not None, "Failed to load metadata from checkpoint"

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from hydra import compose
from hydra import initialize

from anemoi.training.train.train import AnemoiTrainer


def test_training_cycle() -> None:
    initialize(version_base=None, config_path="", job_name="test_training")
    cfg = compose(config_name="config_test")
    AnemoiTrainer(cfg).train()


if __name__ == "__main__":
    test_training_cycle()

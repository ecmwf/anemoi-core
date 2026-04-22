# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from pathlib import Path
from typing import Union

import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.models.migrations import Migrator
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import TemporaryDirectoryForTestData

LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_working_directory() -> None:
    """Automatically set the working directory to the repo root."""
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    os.chdir(repo_root)


def _load_testing_modifications(tmp_path: Path) -> Union[DictConfig, ListConfig]:
    modifications_file = "training/tests/integration/config/testing_modifications.yaml"
    testing_modifications = OmegaConf.load(Path.cwd() / modifications_file)
    assert isinstance(testing_modifications, DictConfig)
    testing_modifications.system.output.root = str(tmp_path)
    return testing_modifications


@pytest.fixture
def testing_modifications_with_temp_dir(tmp_path: Path) -> DictConfig:
    return _load_testing_modifications(tmp_path)


class GetTmpPath:
    def __init__(self, temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> None:
        self.temporary_directory_for_test_data = temporary_directory_for_test_data

    def __call__(self, url: str) -> tuple[str, list[str], list[str]]:

        url_archive = url + ".tgz"
        name_dataset = Path(url).name
        tmp_path_dataset = self.temporary_directory_for_test_data(url_archive, archive=True)

        tmp_path = Path(tmp_path_dataset) / name_dataset

        return tmp_path, url_archive


@pytest.fixture
def get_tmp_path(temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> GetTmpPath:
    return GetTmpPath(temporary_directory_for_test_data)


@pytest.fixture(
    params=[
        ["config_validation=True", "diagnostics.log.mlflow.enabled=True", "diagnostics.log.mlflow.offline=True"],
        ["config_validation=True", "diagnostics.log.mlflow.enabled=False"],
        [
            "config_validation=False",
            "diagnostics.log.mlflow.enabled=True",
            "system.input.graph=null",
            "diagnostics.log.mlflow.offline=True",
        ],
        ["config_validation=False", "diagnostics.log.mlflow.enabled=False", "system.input.graph=null"],
    ],
    ids=["pydantic_MLflow", "pydantic_no_MLflow", "no_pydantic_MLflow", "no_pydantic_no_MLflow"],
)
def gnn_config_mlflow(
    request: pytest.FixtureRequest,
    gnn_config: tuple[DictConfig, str],
) -> tuple[DictConfig, str, str]:
    overrides = request.param
    config, _ = gnn_config

    cfg = OmegaConf.merge(
        config,
        OmegaConf.from_dotlist(overrides),
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def build_global_config(
    overrides: list[str],
    testing_modifications: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str, str]:

    model_architecture = overrides[0].split("=")[1]

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(config_name="config", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_global.yaml")

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    imputer_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/imputer_modifications.yaml")

    OmegaConf.set_struct(template.data, False)
    cfg = OmegaConf.merge(
        template,
        testing_modifications,
        use_case_modifications,
        imputer_modifications,
    )
    OmegaConf.resolve(cfg)

    return cfg, url_dataset, model_architecture


def _configure_multigpu_model_sharding(cfg: DictConfig) -> None:
    cfg.system.hardware.accelerator = "cuda"
    cfg.system.hardware.num_gpus_per_node = 4
    cfg.system.hardware.num_nodes = 1
    cfg.system.hardware.num_gpus_per_model = 2
    cfg.dataloader.read_group_size = 2
    cfg.model.keep_batch_sharded = True
    cfg.dataloader.batch_size.training = 1
    cfg.dataloader.batch_size.validation = 1
    # Some base fixtures call OmegaConf.resolve() before this helper is used,
    # so we must also update the strategy block directly.
    cfg.training.strategy.num_gpus_per_model = 2
    cfg.training.strategy.read_group_size = 2


def _configure_multigpu_ensemble(cfg: DictConfig) -> None:
    cfg.system.hardware.accelerator = "cuda"
    cfg.system.hardware.num_gpus_per_node = 4
    cfg.system.hardware.num_nodes = 1
    cfg.system.hardware.num_gpus_per_model = 2
    cfg.system.hardware.num_gpus_per_ensemble = 4
    cfg.dataloader.read_group_size = 2
    cfg.model.keep_batch_sharded = True
    cfg.dataloader.batch_size.training = 1
    cfg.dataloader.batch_size.validation = 1
    # Some base fixtures call OmegaConf.resolve() before this helper is used,
    # so we must also update the strategy block directly.
    cfg.training.strategy.num_gpus_per_model = 2
    cfg.training.strategy.num_gpus_per_ensemble = 4
    cfg.training.strategy.read_group_size = 2


@pytest.fixture(
    params=[["model=gnn"], ["model=graphtransformer"]],
    ids=["gnn", "graphtransformer"],
)
def global_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str, str]:
    cfg, url, model_architecture = build_global_config(
        request.param,
        testing_modifications_with_temp_dir,
        get_tmp_path,
    )

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2

    OmegaConf.set_struct(cfg.training.scalers.datasets.data, False)
    cfg.training.scalers.datasets.data["output_steps"] = {
        "_target_": "anemoi.training.losses.scalers.TimeStepScaler",
        "norm": "unit-sum",
        "weights": [1.0, 2.0],
    }

    cfg.training.training_loss.datasets.data.scalers = [
        "pressure_level",
        "general_variable",
        "node_weights",
        "output_steps",
    ]
    return cfg, url, model_architecture


@pytest.fixture
def stretched_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):
        template = compose(config_name="stretched")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_forcing_dataset, url_forcing_dataset = get_tmp_path(use_case_modifications.system.input.forcing_dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.forcing_dataset = str(tmp_dir_forcing_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset, url_forcing_dataset]


@pytest.fixture
def stretched_multigpu_config(
    stretched_config: tuple[DictConfig, list[str]],
) -> tuple[DictConfig, list[str]]:
    cfg, urls = stretched_config
    _configure_multigpu_model_sharding(cfg)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, urls


@pytest.fixture
def multidatasets_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_multidatasets"):
        template = compose(config_name="multi")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_multidatasets.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_dataset_b, url_dataset_b = get_tmp_path(use_case_modifications.system.input.dataset_b)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.dataset_b = str(tmp_dir_dataset_b)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2

    return cfg, [url_dataset, url_dataset_b]


@pytest.fixture
def lam_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_lam"):
        template = compose(config_name="lam")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_lam.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_forcing_dataset, url_forcing_dataset = get_tmp_path(use_case_modifications.system.input.forcing_dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.forcing_dataset = str(tmp_dir_forcing_dataset)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset, url_forcing_dataset]


@pytest.fixture
def lam_multigpu_config(
    lam_config: tuple[DictConfig, list[str]],
) -> tuple[DictConfig, list[str]]:
    cfg, urls = lam_config
    _configure_multigpu_model_sharding(cfg)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, urls


@pytest.fixture
def lam_config_with_graph(
    lam_config: tuple[DictConfig, list[str]],
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str]]:
    existing_graph_config = OmegaConf.load(Path.cwd() / "training/src/anemoi/training/config/graph/existing.yaml")
    cfg, urls = lam_config
    cfg.graph = existing_graph_config

    url_graph = "anemoi-integration-tests/training/graphs/lam-graph-2026-02-19.pt"
    cfg.system.input.graph = Path(get_test_data(url_graph))
    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in the lam training cycle test
    cfg.diagnostics.callbacks = []  # remove RolloutEval callback as it is tested in the lam training cycle test
    return cfg, urls


def handle_truncation_matrices(cfg: DictConfig, get_test_data: GetTestData) -> DictConfig:
    url_loss_matrices = cfg.system.input.loss_matrices_path
    tmp_path_loss_matrices = None

    training_losses_cfg = get_multiple_datasets_config(cfg.training.training_loss)
    for dataset_name, training_loss_cfg in training_losses_cfg.items():
        for file in training_loss_cfg.loss_matrices:
            if file is not None:
                tmp_path_loss_matrices = get_test_data(url_loss_matrices + file)
        if tmp_path_loss_matrices is not None:
            cfg.system.input.loss_matrices_path = Path(tmp_path_loss_matrices).parent
            training_loss_cfg.loss_matrices_path = str(Path(tmp_path_loss_matrices).parent)

            cfg.training.validation_metrics.datasets[dataset_name].multiscale.loss_matrices_path = str(
                Path(tmp_path_loss_matrices).parent,
            )
        cfg.training.training_loss.datasets[dataset_name] = training_loss_cfg
    return cfg


@pytest.fixture
def ensemble_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    get_test_data: GetTestData,
) -> tuple[DictConfig, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_ensemble_crps"):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_ensemble_crps.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)

    cfg = handle_truncation_matrices(cfg, get_test_data)
    assert isinstance(cfg, DictConfig)

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2
    return cfg, url_dataset


@pytest.fixture
def ensemble_multigpu_config(
    ensemble_config: tuple[DictConfig, str],
) -> tuple[DictConfig, str]:
    cfg, url = ensemble_config
    _configure_multigpu_ensemble(cfg)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url


@pytest.fixture
def hierarchical_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_hierarchical"):
        template = compose(config_name="hierarchical")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_global.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    cfg.diagnostics.callbacks = []  # remove RolloutEval callback as it is tested in global training cycle test

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset]


@pytest.fixture
def hierarchical_multigpu_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_hierarchical"):
        template = compose(config_name="hierarchical")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_global.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    _configure_multigpu_model_sharding(cfg)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset]


@pytest.fixture
def graphtransformer_multigpu_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    cfg, url, _model_architecture = build_global_config(
        ["model=graphtransformer"],
        testing_modifications_with_temp_dir,
        get_tmp_path,
    )
    _configure_multigpu_model_sharding(cfg)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url


@pytest.fixture
def autoencoder_config(
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_path: GetTmpPath,
) -> tuple[OmegaConf, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_autoencoder"):
        template = compose(config_name="autoencoder")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_autoencoder.yaml")
    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, [url_dataset]


@pytest.fixture
def gnn_config(testing_modifications_with_temp_dir: DictConfig, get_tmp_path: GetTmpPath) -> tuple[DictConfig, str]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_global.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in global training cycle test
    cfg.diagnostics.callbacks = []  # remove RolloutEval callback as it is tested in global training cycle test
    return cfg, url_dataset


@pytest.fixture(
    params=[  # selects different test cases
        "lam",
        "graphtransformer",
        "stretched",
        "ensemble_crps",
        "diffusiontend",
    ],
    ids=[
        "lam",
        "graphtransformer",
        "stretched",
        "ensemble_crps",
        "diffusiontend",
    ],
)
def benchmark_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
    get_test_data: GetTestData,
) -> tuple[OmegaConf, str]:
    test_case = request.param
    base_config = "config"  # which config we start from in anemoi/training/configs/
    # base_config="config" =>  anemoi/training/configs/config.yaml
    # LAM and Stretched need different base configs

    # change configs based on test case
    if test_case == "graphtransformer":
        overrides = ["model=graphtransformer", "graph=multi_scale"]
    elif test_case == "stretched":
        overrides = []
        base_config = "stretched"
    elif test_case == "lam":
        overrides = []
        base_config = "lam"
    elif test_case == "ensemble_crps":
        overrides = ["model=graphtransformer_ens", "graph=multi_scale"]
        base_config = "ensemble_crps"
    elif test_case == "diffusiontend":
        overrides = [
            "model=graphtransformer_diffusiontend",
            "training.training_method=anemoi.training.train.methods.DiffusionTendencyTraining",
        ]
        base_config = "diffusion"
    else:
        msg = f"Error. Unknown benchmark configuration: {test_case}"
        raise ValueError(msg)

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="benchmark"):
        template = compose(config_name=base_config, overrides=overrides)

    # Settings for benchmarking in general (sets atos paths, enables profiling, disables plotting etc)
    base_benchmark_config = OmegaConf.load(Path.cwd() / Path("training/tests/integration/config/benchmark/base.yaml"))
    # Settings for the specific benchmark test case
    use_case_modifications = OmegaConf.load(
        Path.cwd() / f"training/tests/integration/config/benchmark/{test_case}.yaml",
    )
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications, base_benchmark_config)

    cfg.system.output.profiler = Path(cfg.system.output.root + "/" + cfg.system.output.profiler)
    OmegaConf.resolve(cfg)

    if test_case == "ensemble_crps":
        cfg = handle_truncation_matrices(cfg, get_test_data)
    return cfg, test_case


@pytest.fixture(scope="session")
def migrator() -> Migrator:
    return Migrator()


@pytest.fixture
def global_config_with_checkpoint(
    migrator: Migrator,
    global_config: tuple[DictConfig, str, str],
    get_test_data: GetTestData,
) -> tuple[OmegaConf, str]:

    cfg, dataset_url, model_architecture = global_config

    if "gnn" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-gnn-global-2026-03-06.ckpt",
        )
    elif "graphtransformer" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-graphtransformer-global-2026-03-06.ckpt",
        )
    else:
        msg = f"Unknown architecture in config {cfg.model.architecture}"
        raise ValueError(msg)

    _, new_ckpt, _ = migrator.sync(existing_ckpt)

    checkpoint_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root + "/dummy_id")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, checkpoint_dir / "last.ckpt")

    cfg.training.run_id = "dummy_id"
    cfg.training.max_epochs = 3

    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in global training cycle test
    cfg.diagnostics.callbacks = []  # remove RolloutEval callback as it is tested in global training cycle test

    return cfg, dataset_url


@pytest.fixture
def temporal_downscaler_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    """Compose a runnable configuration for the temporal downscaling model with multiple output steps.

    It is based on `temporal_downscaling.yaml` and only patches paths pointing to the
    sample dataset that the tests download locally.
    """
    # No model override here - the template already sets the dedicated
    # temporal downscaling model + task.
    with initialize(
        version_base=None,
        config_path="../../src/anemoi/training/config",
        job_name="test_temporal_downscaler",
    ):
        template = compose(config_name="temporal_downscaler.yaml")

    use_case_modifications = OmegaConf.load(
        Path.cwd() / "training/tests/integration/config/test_temporal_downscaler.yaml",
    )
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url_dataset


@pytest.fixture
def imerg_target_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_filtering"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_filtering.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    OmegaConf.set_struct(template.data, False)  # allow new keys under data (e.g. target)
    OmegaConf.set_struct(
        template.training.training_loss.datasets.data,
        False,
    )  # allow new keys under data (e.g. target)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url_dataset


@pytest.fixture(
    params=[
        [],
        [
            "model=graphtransformer_diffusiontend",
            "training.training_method=anemoi.training.train.methods.DiffusionTendencyTraining",
        ],
    ],
    ids=["diffusion", "diffusiontend"],
)
def diffusion_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_path: GetTmpPath,
) -> tuple[OmegaConf, str]:
    overrides = request.param

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_diffusion"):
        template = compose(config_name="diffusion", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_diffusion.yaml")
    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, url_dataset


@pytest.fixture
def diffusion_multigpu_config(
    diffusion_config: tuple[OmegaConf, str],
) -> tuple[OmegaConf, str]:
    cfg, url = diffusion_config
    _configure_multigpu_model_sharding(cfg)
    OmegaConf.resolve(cfg)
    return cfg, url


@pytest.fixture(
    params=[
        pytest.param(
            [
                "model=graphtransformer_diffusion",
                "training.training_method=anemoi.training.train.methods.DiffusionTraining",
            ],
            id="diffusion",
        ),
        pytest.param(
            [
                "model=graphtransformer_diffusiontend",
                "training.training_method=anemoi.training.train.methods.DiffusionTendencyTraining",
            ],
            id="diffusiontend",
        ),
    ],
    ids=["diffusion", "diffusiontend"],
)
def multidatasets_diffusion_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    overrides = request.param
    is_tendency = any("graphtransformer_diffusiontend" in override for override in overrides)

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_multi_diffusion"):
        template = compose(config_name="multi", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_multidatasets.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_dataset_b, url_dataset_b = get_tmp_path(use_case_modifications.system.input.dataset_b)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.dataset_b = str(tmp_dir_dataset_b)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    if is_tendency:
        cfg.task.multistep_input = 3
        cfg.task.multistep_output = 2
    else:
        cfg.task.multistep_input = 2
        cfg.task.multistep_output = 1
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.diagnostics.plot.callbacks = (
        []
    )  # remove plotting callbacks as they are tested in multidatasets and diffusion test cases
    return cfg, [url_dataset, url_dataset_b]


@pytest.fixture
def mlflow_dry_run_config(gnn_config: tuple[DictConfig, str], mlflow_server: str) -> tuple[DictConfig, str]:
    cfg, url = gnn_config
    cfg["diagnostics"]["log"]["mlflow"]["enabled"] = True
    cfg["diagnostics"]["log"]["mlflow"]["tracking_uri"] = mlflow_server
    cfg["diagnostics"]["log"]["mlflow"]["offline"] = False
    return cfg, url


# ---------------------------------------------------------------------------
# Unified parameterized multigpu fixtures
# ---------------------------------------------------------------------------

_MULTIGPU_MODEL_TYPES = [
    "stretched",
    "lam",
    "graphtransformer",
    "hierarchical",
    "diffusion",
    "diffusiontend",
    "ensemble",
]

_MULTIGPU_ROLLOUT_MODEL_TYPES = [
    "graphtransformer",
    "hierarchical",
    "ensemble",
]


def _build_multigpu_config(
    model_type: str,
    testing_modifications: DictConfig,
    get_tmp_path: GetTmpPath,
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str]]:
    """Build a multigpu config for the given model type.

    Returns (cfg, urls) where urls is always a list of archive URLs.
    """
    config_dir = Path.cwd() / "training/tests/integration/config"

    if model_type == "stretched":
        with initialize(
            version_base=None,
            config_path="../../src/anemoi/training/config",
            job_name="test_stretched_mg",
        ):
            template = compose(config_name="stretched")
        use_case = OmegaConf.load(config_dir / "test_stretched.yaml")
        assert isinstance(use_case, DictConfig)
        tmp_dataset, url_dataset = get_tmp_path(use_case.system.input.dataset)
        tmp_forcing, url_forcing = get_tmp_path(use_case.system.input.forcing_dataset)
        use_case.system.input.dataset = str(tmp_dataset)
        use_case.system.input.forcing_dataset = str(tmp_forcing)
        cfg = OmegaConf.merge(template, testing_modifications, use_case)
        OmegaConf.resolve(cfg)
        _configure_multigpu_model_sharding(cfg)
        OmegaConf.resolve(cfg)
        urls = [url_dataset, url_forcing]

    elif model_type == "lam":
        with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_lam_mg"):
            template = compose(config_name="lam")
        use_case = OmegaConf.load(config_dir / "test_lam.yaml")
        assert isinstance(use_case, DictConfig)
        tmp_dataset, url_dataset = get_tmp_path(use_case.system.input.dataset)
        tmp_forcing, url_forcing = get_tmp_path(use_case.system.input.forcing_dataset)
        use_case.system.input.dataset = str(tmp_dataset)
        use_case.system.input.forcing_dataset = str(tmp_forcing)
        cfg = OmegaConf.merge(template, testing_modifications, use_case)
        OmegaConf.resolve(cfg)
        _configure_multigpu_model_sharding(cfg)
        OmegaConf.resolve(cfg)
        urls = [url_dataset, url_forcing]

    elif model_type == "graphtransformer":
        cfg, url, _ = build_global_config(
            ["model=graphtransformer"],
            testing_modifications,
            get_tmp_path,
        )
        _configure_multigpu_model_sharding(cfg)
        OmegaConf.resolve(cfg)
        urls = [url]

    elif model_type == "hierarchical":
        with initialize(
            version_base=None,
            config_path="../../src/anemoi/training/config",
            job_name="test_hierarchical_mg",
        ):
            template = compose(config_name="hierarchical")
        use_case = OmegaConf.load(config_dir / "test_global.yaml")
        assert isinstance(use_case, DictConfig)
        tmp_dataset, url_dataset = get_tmp_path(use_case.system.input.dataset)
        use_case.system.input.dataset = str(tmp_dataset)
        cfg = OmegaConf.merge(template, testing_modifications, use_case)
        _configure_multigpu_model_sharding(cfg)
        OmegaConf.resolve(cfg)
        urls = [url_dataset]

    elif model_type in ("diffusion", "diffusiontend"):
        overrides = []
        if model_type == "diffusiontend":
            overrides = [
                "model=graphtransformer_diffusiontend",
                "training.model_task=anemoi.training.train.tasks.GraphDiffusionTendForecaster",
            ]
        with initialize(
            version_base=None,
            config_path="../../src/anemoi/training/config",
            job_name="test_diffusion_mg",
        ):
            template = compose(config_name="diffusion", overrides=overrides)
        use_case = OmegaConf.load(config_dir / "test_diffusion.yaml")
        tmp_dataset, url_dataset = get_tmp_path(use_case.system.input.dataset)
        use_case.system.input.dataset = str(tmp_dataset)
        cfg = OmegaConf.merge(template, testing_modifications, use_case)
        OmegaConf.resolve(cfg)
        _configure_multigpu_model_sharding(cfg)
        OmegaConf.resolve(cfg)
        urls = [url_dataset]

    elif model_type == "ensemble":
        overrides = ["model=graphtransformer_ens", "graph=multi_scale"]
        with initialize(
            version_base=None,
            config_path="../../src/anemoi/training/config",
            job_name="test_ensemble_mg",
        ):
            template = compose(config_name="ensemble_crps", overrides=overrides)
        use_case = OmegaConf.load(config_dir / "test_ensemble_crps.yaml")
        assert isinstance(use_case, DictConfig)
        tmp_dataset, url_dataset = get_tmp_path(use_case.system.input.dataset)
        use_case.system.input.dataset = str(tmp_dataset)
        cfg = OmegaConf.merge(template, testing_modifications, use_case)
        OmegaConf.resolve(cfg)
        cfg = handle_truncation_matrices(cfg, get_test_data)
        cfg.training.multistep_input = 3
        cfg.training.multistep_output = 2
        _configure_multigpu_ensemble(cfg)
        OmegaConf.resolve(cfg)
        urls = [url_dataset]

    else:
        msg = f"Unknown multigpu model type: {model_type}"
        raise ValueError(msg)

    assert isinstance(cfg, DictConfig)
    return cfg, urls


@pytest.fixture(
    params=_MULTIGPU_MODEL_TYPES,
    ids=_MULTIGPU_MODEL_TYPES,
)
def multigpu_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str], str]:
    """Parameterized fixture that builds multigpu configs for each model type."""
    model_type = request.param
    cfg, urls = _build_multigpu_config(
        model_type,
        testing_modifications_with_temp_dir,
        get_tmp_path,
        get_test_data,
    )
    return cfg, urls, model_type


@pytest.fixture(
    params=_MULTIGPU_ROLLOUT_MODEL_TYPES,
    ids=_MULTIGPU_ROLLOUT_MODEL_TYPES,
)
def multigpu_rollout_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str], str]:
    """Parameterized fixture for model types that support rollout > 1."""
    model_type = request.param
    cfg, urls = _build_multigpu_config(
        model_type,
        testing_modifications_with_temp_dir,
        get_tmp_path,
        get_test_data,
    )
    return cfg, urls, model_type

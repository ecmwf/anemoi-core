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

    cfg.training.multistep_input = 3
    cfg.training.multistep_output = 2

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

    cfg.training.multistep_input = 3
    cfg.training.multistep_output = 2

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
def lam_config_with_graph(
    lam_config: tuple[DictConfig, list[str]],
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str]]:
    existing_graph_config = OmegaConf.load(Path.cwd() / "training/src/anemoi/training/config/graph/existing.yaml")
    cfg, urls = lam_config
    cfg.graph = existing_graph_config

    dataset_name = "data"  # default dataset name
    url_graph = "anemoi-integration-tests/training/graphs/lam-graph.pt"
    tmp_path_graph = Path(get_test_data(url_graph))
    dataset_graph_filename = tmp_path_graph.name.replace(".pt", f"_{dataset_name}.pt")
    tmp_path_graph.rename(tmp_path_graph.parent / dataset_graph_filename)
    cfg.system.input.graph = tmp_path_graph
    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in lam training cycle test
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

    cfg.training.multistep_input = 3
    cfg.training.multistep_output = 2
    return cfg, url_dataset


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
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset]


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
    return cfg, url_dataset


@pytest.fixture(
    params=[  # selects different test cases
        #"lam",
        #"graphtransformer",
        #"stretched",
        #"ensemble_crps",
        "diffusiontend",
    ],
    ids=[
        #"lam",
        #"graphtransformer",
        #"stretched",
        #"ensemble_crps",
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
        overrides = ["model=graphtransformer_diffusiontend", "training.model_task=anemoi.training.train.tasks.GraphDiffusionTendForecaster",]
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


@pytest.fixture(
    params=[
        ["model=gnn"],
        ["model=graphtransformer"],
    ],
    ids=["gnn", "graphtransformer"],
)
def global_config_with_checkpoint(
    migrator: Migrator,
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    get_test_data: GetTestData,
) -> tuple[OmegaConf, str]:
    # Reuse the same overrides that global_config gets
    overrides = request.param

    cfg, dataset_url, model_architecture = build_global_config(
        overrides,
        testing_modifications_with_temp_dir,
        get_tmp_path,
    )
    # rest of your logic...
    if "gnn" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-gnn-global-2026-01-23.ckpt",
        )
    elif "graphtransformer" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-graphtransformer-global-2026-01-23.ckpt",
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

    return cfg, dataset_url


@pytest.fixture
def multi_output_interpolator_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    """Compose a runnable configuration for the temporal-interpolation model with multiple output steps.

    It is based on `interpolator_multiout.yaml` and only patches paths pointing to the
    sample dataset that the tests download locally.
    """
    # No model override here - the template already sets the dedicated
    # interpolator model + GraphMultiOutInterpolator Lightning task.
    with initialize(
        version_base=None,
        config_path="../../src/anemoi/training/config",
        job_name="test_interpolator_multiout",
    ):
        template = compose(config_name="interpolator_multiout")

    use_case_modifications = OmegaConf.load(
        Path.cwd() / "training/tests/integration/config/test_interpolator_multiout.yaml",
    )
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url_dataset


@pytest.fixture(
    params=[
        [],
        [
            "model=graphtransformer_diffusiontend",
            "training.model_task=anemoi.training.train.tasks.GraphDiffusionTendForecaster",
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


@pytest.fixture(
    params=[
        pytest.param(
            [
                "model=graphtransformer_diffusion",
                "training.model_task=anemoi.training.train.tasks.GraphDiffusionForecaster",
            ],
            id="diffusion",
        ),
        pytest.param(
            [
                "model=graphtransformer_diffusiontend",
                "training.model_task=anemoi.training.train.tasks.GraphDiffusionTendForecaster",
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
        cfg.training.multistep_input = 3
        cfg.training.multistep_output = 2
    else:
        cfg.training.multistep_input = 2
        cfg.training.multistep_output = 1
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

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

import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.migrations import Migrator
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


@pytest.fixture
def testing_modifications_with_temp_dir(tmp_path: Path) -> DictConfig:
    testing_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/testing_modifications.yaml")
    assert isinstance(testing_modifications, DictConfig)
    testing_modifications.hardware.paths.output = str(tmp_path)
    return testing_modifications


class GetTmpPaths:
    def __init__(self, temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> None:
        self.temporary_directory_for_test_data = temporary_directory_for_test_data

    def __call__(self, config: DictConfig, list_datasets: list[str]) -> tuple[str, list[str], list[str]]:
        tmp_paths = []
        dataset_names = []
        archive_urls = []

        for dataset in list_datasets:
            url_archive = config.hardware.files[dataset] + ".tgz"
            name_dataset = Path(config.hardware.files[dataset]).name
            tmp_path_dataset = self.temporary_directory_for_test_data(url_archive, archive=True)

            tmp_paths.append(tmp_path_dataset)
            dataset_names.append(name_dataset)
            archive_urls.append(url_archive)

        if len(list_datasets) == 1:
            return tmp_paths[0], dataset_names, archive_urls

        tmp_dir = os.path.commonprefix([tmp_paths[0], tmp_paths[1]])[:-1]  # remove trailing slash
        rel_paths = [Path(Path(path).name) / name for (name, path) in zip(dataset_names, tmp_paths, strict=False)]
        return tmp_dir, rel_paths, archive_urls


@pytest.fixture
def get_tmp_paths(temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> GetTmpPaths:
    return GetTmpPaths(temporary_directory_for_test_data)


@pytest.fixture(
    params=[
        ["config_validation=True", "diagnostics.log.mlflow.enabled=True", "diagnostics.log.mlflow.offline=True"],
        ["config_validation=True", "diagnostics.log.mlflow.enabled=False"],
        [
            "config_validation=False",
            "diagnostics.log.mlflow.enabled=True",
            "hardware.files.graph=null",
            "diagnostics.log.mlflow.offline=True",
        ],
        ["config_validation=False", "diagnostics.log.mlflow.enabled=False", "hardware.files.graph=null"],
    ],
)
def base_global_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, str, str]:
    overrides = request.param
    model_architecture = overrides[0].split("=")[1]
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(
            config_name="config",
            overrides=overrides,
        )  # apply architecture overrides to template since they override a default
    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    # Add the imputer here as it's not part of the default config
    imputer_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/imputer_modifications.yaml")

    OmegaConf.set_struct(template.data, False)  # allow adding new keys to the template to add the imputer
    cfg = OmegaConf.merge(
        template,
        testing_modifications_with_temp_dir,
        use_case_modifications,
        imputer_modifications,
        OmegaConf.from_dotlist(overrides),
    )

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls[0], model_architecture


@pytest.fixture(
    params=[
        ["model=gnn"],
        ["model=graphtransformer"],
    ],
)
def architecture_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, str, str]:
    overrides = request.param
    model_architecture = overrides[0].split("=")[1]
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(
            config_name="config",
            overrides=overrides,
        )  # apply architecture overrides to template since they override a default

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    # Add the imputer here as it's not part of the default config
    imputer_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/imputer_modifications.yaml")

    OmegaConf.set_struct(template.data, False)  # allow adding new keys to the template to add the imputer
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications, imputer_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls[0], model_architecture


@pytest.fixture
def stretched_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):
        template = compose(config_name="stretched")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset", "forcing_dataset"])
    dataset, forcing_dataset = rel_paths
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = dataset
    use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls


@pytest.fixture
def lam_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_lam"):
        template = compose(config_name="lam")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_lam.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset", "forcing_dataset"])
    dataset, forcing_dataset = rel_paths
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = dataset
    use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls


@pytest.fixture
def lam_config_with_graph(
    lam_config: tuple[DictConfig, list[str]],
    get_test_data: GetTestData,
) -> tuple[DictConfig, list[str]]:
    existing_graph_config = OmegaConf.load(Path.cwd() / "training/src/anemoi/training/config/graph/existing.yaml")
    cfg, urls = lam_config
    cfg.graph = existing_graph_config

    url_graph = cfg.hardware.files["graph"]
    tmp_path_graph = get_test_data(url_graph)
    cfg.hardware.paths.graph = Path(tmp_path_graph).parent
    cfg.hardware.files.graph = Path(tmp_path_graph).name

    return cfg, urls


@pytest.fixture
def ensemble_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_ensemble_crps"):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_ensemble_crps.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls[0]


@pytest.fixture
def hierarchical_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_hierarchical"):
        template = compose(config_name="hierarchical")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls


@pytest.fixture
def gnn_config(testing_modifications_with_temp_dir: DictConfig, get_tmp_paths: GetTmpPaths) -> tuple[DictConfig, str]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls[0]


@pytest.fixture(
    params=[  # selects different test cases
        "lam",
        "graphtransformer",
        "stretched",
        "ensemble_crps",
    ],
)
def benchmark_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
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
        overrides = ["model=graphtransformer", "graph=multi_scale"]
        base_config = "ensemble_crps"
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
    OmegaConf.resolve(cfg)
    return cfg, test_case


@pytest.fixture(scope="session")
def migrator() -> Migrator:
    return Migrator()


@pytest.fixture
def architecture_config_with_checkpoint(
    migrator: Migrator,
    architecture_config: tuple[DictConfig, str, str],
    get_test_data: GetTestData,
) -> tuple[DictConfig, str]:
    cfg, dataset_url, model_architecture = architecture_config

    # Pick checkpoint path based on the model
    if "gnn" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-gnn-global-2025-07-31.ckpt",
        )
    elif "graphtransformer" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-graphtransformer-global-2025-07-31.ckpt",
        )
    else:
        msg_error = f"Unknown architecture in config: {cfg.model.architecture}"
        raise ValueError(msg_error)

    _, new_ckpt, _ = migrator.sync(existing_ckpt)

    checkpoint_dir = Path(cfg.hardware.paths.output + "checkpoint/dummy_id")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, checkpoint_dir / "last.ckpt")

    cfg.training.run_id = "dummy_id"
    cfg.training.max_epochs = 3
    return cfg, dataset_url


@pytest.fixture
def interpolator_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_paths: GetTmpPaths,
) -> tuple[DictConfig, str]:
    """Compose a runnable configuration for the temporal-interpolation model.

    It is based on `interpolator.yaml` and only patches paths pointing to the
    sample dataset that the tests download locally.
    """
    # No model override here - the template already sets the dedicated
    # interpolator model + GraphInterpolator Lightning task.
    with initialize(
        version_base=None,
        config_path="../../src/anemoi/training/config",
        job_name="test_interpolator",
    ):
        template = compose(config_name="interpolator")

    use_case_modifications = OmegaConf.load(
        Path.cwd() / "training/tests/integration/config/test_interpolator.yaml",
    )
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, dataset_urls[0]


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
    get_tmp_paths: callable,
) -> tuple[OmegaConf, str]:
    overrides = request.param

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_diffusion"):
        template = compose(config_name="diffusion", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_diffusion.yaml")
    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls[0]

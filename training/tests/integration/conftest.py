# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
import shutil
from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf


@pytest.fixture(autouse=True)
def set_working_directory() -> None:
    """Automatically set the working directory to the repo root."""
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    os.chdir(repo_root)


@pytest.fixture
def testing_modifications_with_temp_dir(tmp_path: Path) -> OmegaConf:
    testing_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/testing_modifications.yaml")
    testing_modifications.hardware.paths.output = str(tmp_path)
    return testing_modifications


@pytest.fixture(
    params=[
        ["model=gnn"],
        ["model=graphtransformer"],
    ],
)
def architecture_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_paths: callable,
) -> tuple[OmegaConf, str]:
    overrides = request.param
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(
            config_name="config",
            overrides=overrides,
        )  # apply architecture overrides to template since they override a default

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls[0]


@pytest.fixture
def stretched_config(
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_paths: callable,
) -> tuple[OmegaConf, list[str]]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):
        template = compose(config_name="stretched")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset", "forcing_dataset"])
    dataset, forcing_dataset = rel_paths
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = dataset
    use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls


@pytest.fixture
def lam_config(testing_modifications_with_temp_dir: OmegaConf, get_tmp_paths: callable) -> tuple[OmegaConf, str]:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_lam"):
        template = compose(config_name="lam")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_lam.yaml")

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset", "forcing_dataset"])
    dataset, forcing_dataset = rel_paths
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = dataset
    use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls


@pytest.fixture
def lam_config_with_graph(lam_config: OmegaConf, get_test_data: callable) -> tuple[OmegaConf, list[str]]:
    existing_graph_config = OmegaConf.load(Path.cwd() / "training/src/anemoi/training/config/graph/existing.yaml")
    cfg, urls = lam_config
    cfg.graph = existing_graph_config

    url_graph = cfg.hardware.files["graph"]
    tmp_path_graph = get_test_data(url_graph)
    cfg.hardware.paths.graph = Path(tmp_path_graph).parent
    cfg.hardware.files.graph = Path(tmp_path_graph).name

    return cfg, urls


@pytest.fixture
def ensemble_config(testing_modifications_with_temp_dir: OmegaConf, get_tmp_paths: callable) -> tuple[OmegaConf, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_ensemble_crps"):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_ensemble_crps.yaml")

    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls[0]


@pytest.fixture
def get_tmp_paths(temporary_directory_for_test_data: callable) -> callable:
    def _get_tmp_paths(config: OmegaConf, list_datasets: list[str]) -> tuple[str, list[str], list[str]]:
        tmp_paths = []
        dataset_names = []
        archive_urls = []

        for dataset in list_datasets:
            url_archive = config.hardware.files[dataset] + ".tgz"
            name_dataset = Path(config.hardware.files[dataset]).name
            tmp_path_dataset = temporary_directory_for_test_data(url_archive, archive=True)

            tmp_paths.append(tmp_path_dataset)
            dataset_names.append(name_dataset)
            archive_urls.append(url_archive)

        if len(list_datasets) == 1:
            return tmp_paths[0], dataset_names, archive_urls

        tmp_dir = os.path.commonprefix([tmp_paths[0], tmp_paths[1]])[:-1]  # remove trailing slash
        rel_paths = [Path(Path(path).name) / name for (name, path) in zip(dataset_names, tmp_paths)]
        return tmp_dir, rel_paths, archive_urls

    return _get_tmp_paths


@pytest.fixture
def gnn_config(
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_paths: callable,
) -> tuple[OmegaConf, str]:

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")
    tmp_dir, rel_paths, dataset_urls = get_tmp_paths(use_case_modifications, ["dataset"])
    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = rel_paths[0]

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, dataset_urls[0]


@pytest.fixture
def gnn_config_with_checkpoint(gnn_config: OmegaConf, get_test_data: callable) -> OmegaConf:
    cfg, dataset_url = gnn_config
    existing_ckpt = get_test_data(
        "anemoi-integration-tests/training/checkpoints/testing-checkpoint-global-2025-06-24.ckpt",
    )
    checkpoint_dir = Path(cfg.hardware.paths.output + "checkpoint/dummy_id")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(existing_ckpt, checkpoint_dir / "last.ckpt")

    cfg.training.run_id = "dummy_id"
    cfg.training.max_epochs = 3
    return cfg, dataset_url

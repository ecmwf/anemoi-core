# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

from anemoi.models.migrations import Migrator
from anemoi.training.testing import GetTmpPath
from anemoi.utils.testing import GetTestData

CONFIG_DIR = Path(__file__).parent / "config"


@pytest.fixture
def testing_modifications_with_temp_dir(config_with_tempdir: DictConfig) -> DictConfig:
    testing_modifications = OmegaConf.load(CONFIG_DIR / "testing_modifications.yaml")
    testing_modifications_with_tempdir = OmegaConf.merge(testing_modifications, config_with_tempdir)

    assert isinstance(testing_modifications_with_tempdir, DictConfig)
    return testing_modifications_with_tempdir


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


@pytest.fixture
def gnn_config_with_rollout(gnn_config: tuple[DictConfig, str, str]) -> tuple[DictConfig, str, str]:
    cfg, url_dataset = gnn_config
    cfg.task.rollout = {
        "start": 1,
        "epoch_increment": 1,
        "maximum": 4,
    }
    return cfg, url_dataset


def build_global_config(
    overrides: list[str],
    testing_modifications: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str, str]:

    model_architecture = overrides[0].split("=")[1]

    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="test_config"):
        template = compose(config_name="config", overrides=overrides)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_global.yaml")

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    imputer_modifications = OmegaConf.load(CONFIG_DIR / "imputer_modifications.yaml")

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
    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="test_stretched"):
        template = compose(config_name="stretched")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_stretched.yaml")
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
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_multidatasets",
    ):
        template = compose(config_name="multi")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_multidatasets.yaml")
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
    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="test_lam"):
        template = compose(config_name="lam")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_lam.yaml")
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

    url_graph = "anemoi-integration-tests/training/graphs/lam-graph-2026-02-19.pt"
    cfg.system.input.graph = Path(get_test_data(url_graph))
    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in the lam training cycle test
    cfg.diagnostics.callbacks = []  # no extra callbacks
    return cfg, urls


@pytest.fixture
def ensemble_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    handle_truncation_matrices: Callable[[DictConfig], DictConfig],
) -> tuple[DictConfig, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_ensemble_crps",
    ):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_ensemble_crps.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)

    cfg = handle_truncation_matrices(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2
    return cfg, url_dataset


@pytest.fixture
def ensemble_graph_multiscale_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_ensemble_graph",
    ):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_ensemble_crps.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    cfg.training.training_loss.datasets.data.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.training.training_loss.datasets.data.multiscale_config = {
        "num_scales": 4,
        "base_num_nearest_neighbours": 16,
        "base_sigma": 0.01570,
        "scale_factor": 2,
    }

    cfg.training.validation_metrics.datasets.data.multiscale.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.training.validation_metrics.datasets.data.multiscale.multiscale_config = {
        "num_scales": 4,
        "base_num_nearest_neighbours": 16,
        "base_sigma": 0.01570,
        "scale_factor": 2,
    }

    cfg.diagnostics.plot.callbacks = []
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2
    return cfg, url_dataset


@pytest.fixture
def ensemble_truncated_connection_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    overrides = ["model=graphtransformer_ens", "graph=multi_scale"]

    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_ensemble_truncated_connection",
    ):
        template = compose(config_name="ensemble_crps", overrides=overrides)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_ensemble_crps.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    cfg.training.training_loss.datasets.data.multiscale_config = None
    cfg.training.training_loss.datasets.data.weights = [1.0]

    cfg.training.validation_metrics.datasets.data.multiscale.multiscale_config = None
    cfg.training.validation_metrics.datasets.data.multiscale.weights = [1.0]
    cfg.diagnostics.plot.callbacks = []

    cfg.model.residual = OmegaConf.create(
        {
            "_target_": "anemoi.models.layers.residual.TruncatedConnection",
            "truncation_config": {
                "grid": "o32",
                "num_nearest_neighbours": 3,
                "sigma": 1.0,
            },
        },
    )

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.task.multistep_input = 3
    cfg.task.multistep_output = 2
    return cfg, url_dataset


@pytest.fixture
def hierarchical_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_hierarchical",
    ):
        template = compose(config_name="hierarchical")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_global.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    cfg.diagnostics.callbacks = []  # no extra callbacks

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, [url_dataset]


@pytest.fixture
def autoencoder_config(
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_path: GetTmpPath,
) -> tuple[OmegaConf, list[str]]:
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_autoencoder",
    ):
        template = compose(config_name="autoencoder")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_autoencoder.yaml")
    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, [url_dataset]


@pytest.fixture
def gnn_config(testing_modifications_with_temp_dir: DictConfig, get_tmp_path: GetTmpPath) -> tuple[DictConfig, str]:
    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="test_config"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_global.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in global training cycle test
    cfg.diagnostics.callbacks = []  # no extra callbacks
    return cfg, url_dataset


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
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-gnn-global-2026-08-18.ckpt",
        )
    elif "graphtransformer" in model_architecture:
        existing_ckpt = get_test_data(
            "anemoi-integration-tests/training/checkpoints/testing-checkpoint-graphtransformer-global-2026-08-18.ckpt",
        )
    else:
        msg = f"Unknown architecture in config {cfg.model.architecture}"
        raise ValueError(msg)

    _, new_ckpt, _ = migrator.sync(existing_ckpt)

    checkpoint_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root + "/dummy_id")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, checkpoint_dir / "last.ckpt")

    # Resume the staged run via the checkpoint pipeline surface (the legacy
    # ``training.run_id`` key was removed): a RunIdSource (fork=false) resolves the
    # run's last.ckpt and a WarmStartLoader restores it.
    with open_dict(cfg):
        cfg.training.checkpoint = {
            "source": {
                "_target_": "anemoi.training.checkpoint.sources.run.RunIdSource",
                "run_id": "dummy_id",
                "fork": False,
            },
            "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
        }
    cfg.training.max_epochs = 3

    cfg.diagnostics.plot.callbacks = []  # remove plotting callbacks as they are tested in global training cycle test
    cfg.diagnostics.callbacks = []  # no extra callbacks

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
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_temporal_downscaler",
    ):
        template = compose(config_name="temporal_downscaler_ensemble.yaml")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_temporal_downscaler.yaml")
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
    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="test_filtering"):
        template = compose(config_name="config")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_filtering.yaml")
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
    params=["transport_edm_diffusion", "transport_edm_diffusion_tendency"],
    ids=["transport_edm_diffusion", "transport_edm_diffusion_tendency"],
)
def edm_transport_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_path: GetTmpPath,
) -> tuple[OmegaConf, str]:
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_edm_transport",
    ):
        template = compose(config_name=request.param)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_transport.yaml")
    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, url_dataset


@pytest.fixture(
    params=["transport_stochastic_interpolant", "transport_stochastic_interpolant_tendency"],
    ids=["transport_stochastic_interpolant", "transport_stochastic_interpolant_tendency"],
)
def stochastic_interpolant_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
    get_tmp_path: GetTmpPath,
) -> tuple[OmegaConf, str]:
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_stochastic_interpolant",
    ):
        template = compose(config_name=request.param)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_transport.yaml")
    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    return cfg, url_dataset


@pytest.fixture(
    params=[
        pytest.param(
            [
                "model=graphtransformer_multi_transport_edm",
                "training=multi_transport",
                "training.transport.prediction_mode=state",
                "training.transport.objective=edm_diffusion",
            ],
            id="edm_diffusion",
        ),
        pytest.param(
            [
                "model=graphtransformer_multi_transport_tendency_edm",
                "training=multi_transport",
                "training.transport.prediction_mode=tendency",
                "training.transport.objective=edm_diffusion",
            ],
            id="edm_diffusion_tendency",
        ),
    ],
    ids=["edm_diffusion", "edm_diffusion_tendency"],
)
def multidatasets_edm_transport_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, list[str]]:
    overrides = request.param
    is_tendency = any("graphtransformer_transport_tendency" in override for override in overrides)

    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_multi_edm_transport",
    ):
        template = compose(config_name="multi", overrides=overrides)

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_multidatasets.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_dataset_b, url_dataset_b = get_tmp_path(use_case_modifications.system.input.dataset_b)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.dataset_b = str(tmp_dir_dataset_b)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    cfg.model.compile = []  # TODO(cathal): debug compile + checkpoint error
    if is_tendency:
        cfg.task.multistep_input = 3
        cfg.task.multistep_output = 2
    else:
        cfg.task.multistep_input = 2
        cfg.task.multistep_output = 1
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    cfg.diagnostics.plot.callbacks = []  # tested in the multidatasets / EDM transport cases
    return cfg, [url_dataset, url_dataset_b]


@pytest.fixture
def mlflow_dry_run_config(gnn_config: tuple[DictConfig, str], mlflow_server: str) -> tuple[DictConfig, str]:
    cfg, url = gnn_config
    cfg["diagnostics"]["log"]["mlflow"]["enabled"] = True
    cfg["diagnostics"]["log"]["mlflow"]["tracking_uri"] = mlflow_server
    cfg["diagnostics"]["log"]["mlflow"]["offline"] = False
    return cfg, url


@pytest.fixture
def temporal_downscaler_ensemble_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
    handle_truncation_matrices: Callable[[DictConfig], DictConfig],
) -> tuple[DictConfig, str]:

    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_temporal_downscaler_ensemble",
    ):
        template = compose(config_name="temporal_downscaler_ensemble")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_temporal_downscaler_ensemble.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)

    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)

    cfg = handle_truncation_matrices(cfg)
    assert isinstance(cfg, DictConfig)

    return cfg, url_dataset


@pytest.fixture
def offset_forecaster_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    cfg, url, _ = build_global_config(
        ["model=graphtransformer"],
        testing_modifications_with_temp_dir,
        get_tmp_path,
    )

    OmegaConf.set_struct(cfg.task, False)
    cfg.task = {
        "_target_": "anemoi.training.tasks.OffsetForecaster",
        "input_offsets": ["-12H", "0H"],
        "output_offsets": ["6H", "12H"],
        "rollout": {
            "start": 2,
            "epoch_increment": 0,
            "maximum": 2,
        },
    }

    return cfg, url


@pytest.fixture
def offset_forecaster_tendency_transport_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path: GetTmpPath,
) -> tuple[DictConfig, str]:
    """Compose a multi-output tendency transport model using irregular input offsets."""
    with initialize_config_module(
        version_base=None,
        config_module="anemoi.training.config",
        job_name="test_offset_forecaster_tendency_transport",
    ):
        template = compose(config_name="transport_edm_diffusion_tendency")

    use_case_modifications = OmegaConf.load(CONFIG_DIR / "test_transport.yaml")
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)

    OmegaConf.set_struct(cfg.task, False)
    cfg.task = {
        "_target_": "anemoi.training.tasks.OffsetForecaster",
        "input_offsets": ["-12H", "0H"],
        "output_offsets": ["6H", "12H"],
        "rollout": {
            "start": 1,
            "epoch_increment": 0,
            "maximum": 1,
        },
        "validation_rollout": 1,
    }
    cfg.training.max_epochs = 1
    cfg.dataloader.limit_batches.training = 1
    cfg.dataloader.limit_batches.validation = 1
    cfg.diagnostics.plot.callbacks = []

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg, url_dataset


@pytest.fixture(scope="session")
def partial_metadata_schema() -> dict[str, Any]:
    """Defines the expected structure of the metadata dictionary produced by the trainer.

    We only check a partial schema here, since we are in the process of consolidating the metadata structure
    and content.
    The goal is to prevent regressions in the metadata structure we are establishing for anemoi-inference
    while allowing flexibility in the rest of the metadata content and structure as we iterate towards
    a complete schema.
    After the consolidation is complete, we can migrate to a complete schema and potentially
    use pydantic for validation.
    Before making changes to the partial schema below, check whether the change is compatible with anemoi-inference.
    """
    return {
        "version": None,
        "config": None,
        "seed": None,
        "base_seed": None,
        "run_id": None,
        "dataset": None,
        "data_indices": None,
        "provenance_training": None,
        "timestamp": None,
        "metadata_inference": {
            "seed": None,
            "base_seed": None,
            "run_id": None,
            "dataset_names": list,  # list of datasets
            "task": None,
            "__datasets__": {  # schema applied to each dataset entry
                "timesteps": {
                    "relative_date_indices_training": None,
                    "input_relative_date_indices": None,
                    "output_relative_date_indices": None,
                    "timestep": None,
                    # extra entries populated by offset-forecaster only
                    "input_offsets": None,
                    "output_offsets": None,
                    "rollout_shift": None,
                    "advance_map": None,
                },
                "data_indices": {
                    "input": None,
                    "output": None,
                },
                "variable_types": {
                    "forcing": None,
                    "target": None,
                    "prognostic": None,
                    "diagnostic": None,
                },
                "shapes": {
                    "variables": None,
                    "input_timesteps": None,
                    "ensemble": None,
                    "grid": None,
                },
            },
        },
    }

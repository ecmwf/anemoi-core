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

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent / "config"


@pytest.fixture(
    params=[  # selects different test cases
        "lam",
        "graphtransformer",
        "stretched",
        "ensemble_crps",
        "edm_diffusion_tendency",
        "temporal_downscaler_ensemble",
    ],
    ids=[
        "lam",
        "graphtransformer",
        "stretched",
        "ensemble_crps",
        "edm_diffusion_tendency",
        "temporal_downscaler_ensemble",
    ],
)
def benchmark_config(
    request: pytest.FixtureRequest,
    config_with_tempdir: OmegaConf,
    handle_truncation_matrices: Callable[[DictConfig], DictConfig],
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
    elif test_case == "edm_diffusion_tendency":
        overrides = []
        base_config = "transport_edm_diffusion_tendency"
    elif test_case == "temporal_downscaler_ensemble":
        overrides = []
        base_config = "temporal_downscaler_ensemble"
    else:
        msg = f"Error. Unknown benchmark configuration: {test_case}"
        raise ValueError(msg)

    with initialize_config_module(version_base=None, config_module="anemoi.training.config", job_name="benchmark"):
        template = compose(config_name=base_config, overrides=overrides)

    # Settings for benchmarking in general (sets atos paths, enables profiling, disables plotting etc)
    base_benchmark_config = OmegaConf.load(CONFIG_DIR / "base.yaml")
    # Settings for the specific benchmark test case
    use_case_modifications = OmegaConf.load(CONFIG_DIR / f"{test_case}.yaml")
    OmegaConf.set_struct(template.data, False)
    cfg = OmegaConf.merge(template, config_with_tempdir, use_case_modifications, base_benchmark_config)

    cfg.system.output.profiler = Path(cfg.system.output.root + "/" + cfg.system.output.profiler)
    OmegaConf.resolve(cfg)

    if test_case == "ensemble_crps":
        cfg = handle_truncation_matrices(cfg)
    return cfg, test_case

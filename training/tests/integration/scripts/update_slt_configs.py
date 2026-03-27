# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: T201
# allow prints for logging purposes in this script
# ruff: noqa: S603, S607
# allow subprocess calls to git

import argparse
import subprocess
from datetime import UTC
from datetime import datetime
from pathlib import Path

from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf

INTEGRATION_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Update the training configs used in the system-level tests. "
    "Running this script will replace the configs in the system-level test suite in the Anemoi repository.",
)
parser.add_argument(
    "anemoi_repo_root",
    type=str,
    help="Path to the root of the Anemoi repository.",
)
args = parser.parse_args()
anemoi_repo_root = Path(args.anemoi_repo_root).resolve()


def validate_anemoi_repo(root: Path) -> Path:
    if not root.exists():
        msg = f"Provided path does not exist: {root}"
        raise ValueError(msg)

    if not (root / ".git").exists():
        msg = f"{root} does not look like a git repository"
        raise ValueError(msg)

    expected_path = root / "tests/system-level/anemoi_test/configs"
    if not expected_path.exists():
        msg = f"{root} does not appear to be the Anemoi repo (missing {expected_path})"
        raise ValueError(msg)

    return expected_path


def generate_global_config(hydra_config_path: Path) -> OmegaConf:
    with initialize(version_base=None, config_path=hydra_config_path, job_name="test_config"):
        template = compose(config_name="config", overrides=["model=graphtransformer"])

    config_path = INTEGRATION_ROOT / "config" / "test_global.yaml"
    use_case_modifications = OmegaConf.load(config_path)
    use_case_modifications.diagnostics.plot.callbacks = []

    name_dataset = Path(use_case_modifications.system.input.dataset).name
    use_case_modifications.system.input.dataset = "${system.input.root}/" + str(name_dataset)

    testing_modifications = OmegaConf.load(INTEGRATION_ROOT / "config/testing_modifications.yaml")
    imputer_modifications = OmegaConf.load(INTEGRATION_ROOT / "config/imputer_modifications.yaml")

    OmegaConf.set_struct(template.data, False)
    cfg = OmegaConf.merge(
        template,
        testing_modifications,
        use_case_modifications,
        imputer_modifications,
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

    OmegaConf.set_struct(cfg.system.input, False)
    cfg.system.input.root = "dummy_root"  # will be replaced by the actual root in the system-level test suite

    return cfg


def generate_lam_config(hydra_config_path: Path) -> OmegaConf:
    with initialize(version_base=None, config_path=hydra_config_path, job_name="test_config"):
        template = compose(config_name="lam")

    use_case_modifications = OmegaConf.load(INTEGRATION_ROOT / "config/test_lam.yaml")
    use_case_modifications.diagnostics.plot.callbacks = []

    name_dataset = Path(use_case_modifications.system.input.dataset).name
    name_forcing_dataset = Path(use_case_modifications.system.input.forcing_dataset).name
    use_case_modifications.system.input.dataset = "${system.input.root}/" + str(name_dataset)
    use_case_modifications.system.input.forcing_dataset = "${system.input.root}/" + str(name_forcing_dataset)

    testing_modifications = OmegaConf.load(INTEGRATION_ROOT / "config/testing_modifications.yaml")
    cfg = OmegaConf.merge(
        template,
        testing_modifications,
        use_case_modifications,
    )
    OmegaConf.set_struct(cfg.system.input, False)
    cfg.system.input.root = "dummy_root"  # will be replaced by the actual root in the system-level test suite

    return cfg


def git(cmd: list[str], repo_path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_path), *cmd],
        text=True,
        stderr=subprocess.DEVNULL,  # suppress noisy errors
    ).strip()


def get_git_info(repo_path: Path) -> tuple[str, str, bool]:
    try:
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"], repo_path)
        commit = git(["rev-parse", "HEAD"], repo_path)
        dirty = bool(git(["status", "--porcelain"], repo_path))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[WARN] Could not get git info: {e}")
        branch, commit, dirty = "unknown", "unknown", False

    return branch, commit, dirty


def save_with_header(cfg: OmegaConf, path: Path, metadata: dict) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)

    header_lines = [
        "# =========================================",
        "# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        "# =========================================",
    ]

    for k, v in metadata.items():
        header_lines.append(f"# {k}: {v}")

    header_lines.append(f"# generated_at: {datetime.now(tz=UTC).isoformat()}")
    header_lines.append("# =========================================\n")

    with Path.open(path, "w") as f:
        f.write("\n".join(header_lines))
        f.write(yaml_str)


system_level_configs_path = validate_anemoi_repo(anemoi_repo_root)

anemoi_core_repo_path = INTEGRATION_ROOT.parent.parent.parent
branch, commit, dirty = get_git_info(anemoi_core_repo_path)

print("=== Anemoi-core repo state ===")
print(f"Branch : {branch}")
print(f"Commit : {commit}")
print(f"Uncommitted changes  : {dirty}")
print("===========================")

metadata = {
    "source_repo": "anemoi-core",
    "branch": branch,
    "commit": commit,
    "uncommitted_changes": dirty,
}


hydra_config_path = "../../src/anemoi/training/config"
global_config = generate_global_config(hydra_config_path)
lam_config = generate_lam_config(hydra_config_path)


save_with_header(
    global_config,
    system_level_configs_path / "training/global/training_config.yaml",
    metadata,
)

save_with_header(
    lam_config,
    system_level_configs_path / "training/lam/training_config.yaml",
    metadata,
)

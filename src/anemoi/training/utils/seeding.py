import os


def get_base_seed(env_var_list=("ANEMOI_BASE_SEED", "SLURM_JOB_ID")) -> int:
    """Gets the base seed from the environment variables.

    Option to manually set a seed via export ANEMOI_BASE_SEED=xxx in job script
    """
    base_seed = None
    for env_var in env_var_list:
        if env_var in os.environ:
            base_seed = int(os.environ.get(env_var))
            break

    assert base_seed is not None, f"Base seed not found in environment variables {env_var_list}"

    if base_seed < 1000:
        base_seed = base_seed * 1000  # make it (hopefully) big enough

    return base_seed

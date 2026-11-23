#!/usr/bin/env bash
# Local run of the "nightly-integration-tests-hpc-gpu" workflow (.github/workflows/integration-tests-hpc.yml).
# Usage: tools/run-integration-tests.sh [test ...]
#   test: training_cycle, accuracy or benchmark (default: training_cycle accuracy).
# Run this script on a node with a GPU, for example in an salloc or sbatch job.
# The benchmark test needs a Slurm allocation with 2 GPUs and 2 tasks (it uses srun).
# Environment:
#   MLFLOW_TEST_URL        MLflow test server. If it is not set, the MLflow tests are skipped.
#   DATASET_ROOT_PATH      Directory that contains the datasets. The accuracy test needs it.
#   ACCURACY_REFERENCE_ID  MLflow run ID for the accuracy test (empty: hard-coded reference).
#   PYTHON_VERSION         Python version of the venv (default: 3.12, as in CI).
#   TORCH_VERSION          torch minor version, for example 2.10 (default: latest that pyproject allows).
set -euxo pipefail
cd "$(dirname "$0")/.."

tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(training_cycle accuracy)

banner() {
    { set +x; } 2>/dev/null
    local rule; rule=$(printf '%.0s#' {1..72})
    printf '\n%s\n%s\n%s\n\n' "${rule}" "$*" "${rule}" >&2
    set -x
}

# Find incorrect arguments and missing settings before the long venv build starts.
for t in "${tests[@]}"; do
    case ${t} in
        training_cycle) ;;
        accuracy)
            if [ -z "${DATASET_ROOT_PATH:-}" ]; then
                echo "The accuracy test needs DATASET_ROOT_PATH" >&2; exit 1
            fi ;;
        benchmark)
            if [ -z "${SLURM_JOB_ID:-}" ]; then
                echo "The benchmark test needs a Slurm allocation with 2 GPUs and 2 tasks" >&2; exit 1
            fi ;;
        *) echo "Unknown test '${t}'. Use training_cycle, accuracy or benchmark." >&2; exit 1 ;;
    esac
done
command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: nvidia-smi not found, the GPU tests can fail" >&2

mlflow_args=()
if [ -n "${MLFLOW_TEST_URL:-}" ]; then
    mlflow_args=(--mlflow "--mlflow-server=${MLFLOW_TEST_URL}")
else
    banner "MLFLOW_TEST_URL is not set: the MLflow tests are skipped"
fi

unset PYTHONPATH

tmp=$(mktemp -d)
banner "The venv goes to ${tmp}
It is removed only if all tests pass; on failure it is kept for inspection."

venv=${tmp}/venv
export UV_TORCH_BACKEND=auto
uv venv --python "${PYTHON_VERSION:-3.12}" "${venv}"
uv pip install --python "${venv}" \
    -e "./training[all,tests,profile]" -e "./models[all,tests]" -e "./graphs[all,tests]" \
    ${TORCH_VERSION:+"torch==${TORCH_VERSION}.*"}
# PyG 2.8+ uses pyg-lib for accelerated graph construction.
torch_installed=$("${venv}/bin/python" -c "import torch; print(torch.__version__)")
uv pip install --python "${venv}" pyg-lib -f "https://data.pyg.org/whl/torch-${torch_installed}.html"

export PATH="${venv}/bin:${PATH}" VIRTUAL_ENV="${venv}"
failed=()
for t in "${tests[@]}"; do
    banner "Test: ${t}"
    # -n 0: pytest.ini sets -n logical, which starts one xdist worker per CPU, all on the same GPU.
    case ${t} in
        training_cycle)
            python -m pytest -n 0 -v training/tests/integration/training_workflows/ --slow "${mlflow_args[@]}" &&
            python -m pytest -n 0 -v models/tests/integration/ --slow &&
            python -m pytest -n 0 -v graphs/tests/integration/ || failed+=("${t}") ;;
        accuracy)
            python -m pytest -n 0 -v training/tests/integration/accuracy/ --slow "${mlflow_args[@]}" \
                "--accuracy-reference-id=${ACCURACY_REFERENCE_ID:-}" || failed+=("${t}") ;;
        benchmark)
            srun --ntasks-per-node=2 python -m pytest -o tmp_path_retention_policy=all -v \
                training/tests/integration/benchmark/ --slow --multigpu || failed+=("${t}") ;;
    esac
done

if [ ${#failed[@]} -ne 0 ]; then
    banner "FAILED: ${failed[*]}. The venv is kept in ${venv}"
    exit 1
fi
rm -rf "${tmp}"
banner "All integration tests passed; removed ${tmp}"

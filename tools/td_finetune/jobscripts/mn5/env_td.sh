# Environment for the TD runs on MN5: a COPY of the shared AIFS v3 venv (the original is used by the ens-crps/single-mse
# variants and is not modified) with anemoi-training installed from the branch feat/td-imerg-finetune
# (0.17.1 + loss prefilter, version 0.17.1+tdimerg.*) and anemoi-datasets 0.5.42, anemoi-utils 0.5.12 installed on top.
module purge
module load intel/2025.2 impi/2021.16 mkl/2023.2.0
module load hdf5 python/3.12.1
unset PYTHONPATH PYTHONHOME
export TD=/gpfs/scratch/ehpc01/ecm807005/td_mn5
export VENV=$TD/venv
export PATH=$VENV/bin:$PATH
export LD_LIBRARY_PATH=/apps/ACC/PYTHON/3.12.1/INTEL/lib:$LD_LIBRARY_PATH
export MLFLOW_ALLOW_FILE_STORE=true

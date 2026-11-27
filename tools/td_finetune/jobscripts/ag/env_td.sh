# Environment for the TD runs on ag (GH200, aarch64): own venv built from the aifs-dev atos-hoppers lockfile
# (torch 2.9+cu129, anemoi-models 0.19.0, graphs 0.9.7, ...) with anemoi-training installed from the branch
# feat/td-imerg-finetune (0.17.1 + loss prefilter, version 0.17.1+tdimerg.*) and anemoi-datasets 0.5.42, anemoi-utils 0.5.12.
# No PYTHONPATH overlay needed any more.
export TD=/scratch/momc/td_ag
export VENV=$TD/venv
export PATH=$VENV/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/apps/gcc/15.1.0/lib64/:$LD_LIBRARY_PATH
unset PYTHONPATH
export MLFLOW_ALLOW_FILE_STORE=true

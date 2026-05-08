module load Stages/2025  GCCcore/.13.3.0
module load Python/3.12.3

source  /e/data1/jureap-data/ecmwf/users/clare1/time_interpolator_env/bin/activate

anemoi-training mlflow sync -r 9d4481b431c94645942630bad8f40ab1 -s /e/scratch/gkpdm/clare1/new/logs/mlflow/ -d https://mlflow.ecmwf.int -e time_interpolator -a

anemoi-training mlflow sync -r e11804a269a4405aa2095bd6c3c721f9 -s /e/scratch/gkpdm/clare1/logs/mlflow/ -d https://mlflow.ecmwf.int -e time_interpolator -a
anemoi-training mlflow sync -r 7443938cbaf9419685fab06ba9f7abb3 -s /e/scratch/gkpdm/clare1/logs/mlflow/ -d https://mlflow.ecmwf.int -e time_interpolator -a

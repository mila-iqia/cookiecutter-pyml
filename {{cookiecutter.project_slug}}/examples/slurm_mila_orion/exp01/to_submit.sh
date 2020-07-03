#!/bin/bash
# __TODO__ fix options if needed
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user={{ cookiecutter.email }}

export MLFLOW_TRACKING_URI='../mlruns'
export ORION_DB_ADDRESS='../orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config ../orion_config.yaml ../../../{{cookiecutter.project_slug}}/main.py \
    --data ../data --config ../config.yaml --disable-progressbar \
    --output '{exp.working_dir}/{exp.name}_{trial.id}/' \
    --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

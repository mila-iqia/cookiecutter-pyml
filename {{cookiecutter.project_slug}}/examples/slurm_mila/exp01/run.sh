#!/bin/bash
# __TODO__ fix options if needed
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name={{ cookiecutter.project_slug }}
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user={{ cookiecutter.email }}

export MLFLOW_TRACKING_URI='../mlruns'

main --data data --output output --config ../config.yaml
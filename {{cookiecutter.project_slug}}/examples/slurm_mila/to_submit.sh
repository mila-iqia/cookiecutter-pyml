#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name={{ cookiecutter.project_slug }}
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log
# to attach a tag to your run (e.g., used to track the GPU time)
# uncomment the following line and add replace `my_tag` with the proper tag:
##SBATCH --wckey=my_tag
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user={{ cookiecutter.email }}

export MLFLOW_TRACKING_URI='mlruns'

main --data ../data --output output --config config.yaml --disable-progressbar

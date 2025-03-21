#!/bin/bash
{%- if cookiecutter.environment == 'mila' %}
## this is for the mila cluster (uncomment it if you need it):
##SBATCH --account=rrg-bengioy-ad
## this instead for ComputCanada (uncomment it if you need it):
##SBATCH --partition=long
# to attach a tag to your run (e.g., used to track the GPU time)
# uncomment the following line and add replace `my_tag` with the proper tag:
##SBATCH --wckey=my_tag
{%- endif %}
{%- if cookiecutter.environment == 'generic' %}
## set --account=... or --partition=... as needed.
{%- endif %}
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name=amlrt_project
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# to attach a tag to your run (e.g., used to track the GPU time)
# uncomment the following line and add replace `my_tag` with the proper tag:
##SBATCH --wckey=my_tag
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=amlrt_email@mila.quebec

export MLFLOW_TRACKING_URI='mlruns'

amlrt_project_train --data ../data --output output --config ../config.yaml config.yaml --tmp-folder ${SLURM_TMPDIR} --disable-progressbar

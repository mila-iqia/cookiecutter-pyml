#!/bin/bash
#SBATCH --job-name=amlrt_project
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
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=amlrt_email@mila.quebec

export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

amlrt_project_merge_configs --config ../config.yaml config.yaml --merged-config-file merged_config.yaml
orion -v hunt --config orion_config.yaml \
    amlrt_project_train --data ../data --config merged_config.yaml --disable-progressbar \
    --output '{exp.working_dir}/{trial.id}/' \
    --log '{exp.working_dir}/{trial.id}/exp.log' \
    --tmp-folder ${SLURM_TMPDIR}/{trial.id}

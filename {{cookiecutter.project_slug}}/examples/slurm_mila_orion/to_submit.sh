#!/bin/bash
# __TODO__ fix options if needed
#SBATCH --job-name={{ cookiecutter.project_slug }}
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# to attach a tag to your run (e.g., used to track the GPU time)
# uncomment the following line and add replace `my_tag` with the proper tag:
##SBATCH --wckey=my_tag
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user={{ cookiecutter.email }}

export MLFLOW_TRACKING_URI='mlruns'
export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config orion_config.yaml \
    main --data ../data --config config.yaml --disable-progressbar \
    --output '{exp.working_dir}/{exp.name}_{trial.id}/' \
    --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log' \
    --tmp-folder ${SLURM_TMPDIR}

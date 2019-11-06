#!/bin/bash
# __TODO__ fix options if needed
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name={{ cookiecutter.project_slug }}
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log

export MLFLOW_TRACKING_URI='../mlruns'
export ORION_DB_ADDRESS='../orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

# 2. Copy your dataset on the compute node
#export DATADIR=$SLURM_TMPDIR/dataset
#time rsync -a --info=progress2 /__TODO__/path/to/dataset.tar $SLURM_TMPDIR/
#time tar xf $SLURM_TMPDIR/dataset.tar -C $SLURM_TMPDIR/ --strip=4

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
main --data no_data --output no_output --config_general ../config.yaml

# 4. Copy whatever you want to save on $SCRATCH
# rsync -avz $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/

#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --job-name=__TODO__
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log

# 1. Create your environment
module load python/{{ cookiecutter.python_version }}
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cp /__TODO__/path/to/requirements.txt $SLURM_TMPDIR/requirements.txt
# If you need to install from wheels, add the line below
# sed -i '1 i\-f /__TODO__/path/to/wheels' $SLURM_TMPDIR/requirements.txt

pip install --no-index -r $SLURM_TMPDIR/requirements.txt

export ORION_DB_ADDRESS=__TODO__
export ORION_DB_TYPE=__TODO__
export ORION_DB_NAME=__TODO__

# 2. Copy your dataset on the compute node
export DATADIR=$SLURM_TMPDIR/dataset
time rsync -a --info=progress2 /__TODO__/path/to/dataset.tar $SLURM_TMPDIR/
time tar xf $SLURM_TMPDIR/dataset.tar -C $SLURM_TMPDIR/ --strip=4

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd /__TODO__/path/to/code
python -u main.py --data $DATADIR --output no_output --config_general config.yaml

# 4. Copy whatever you want to save on $SCRATCH
# rsync -avz $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/

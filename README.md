Quickstart
----------

Install the latest Cookiecutter if you haven't installed it yet (this requires
Cookiecutter 1.4.0 or higher)::

    pip install -U cookiecutter

Generate a Python package project::

    cookiecutter https://github.com/mirkobronzi/cookiecutter-pyml.git

Then cd into the (new) project folder:

    cd $YOUR_PROJECT_NAME

Add git:

    git init

Go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project.

Add new dependencies (if needed) to setup.py.

Install the dependencies:

    pip install -e .

Change the code as appropriate.
Note that the code should already compile at this point.
Try to run the examples under $ROOT/examples
(see next sections).

Examples
--------

The `examples` folder contains several execution examples.
There is a folder for running without orion and with orion.
Also, there is an example for the following environments:
* local (e.g., you laptop)
* beluga cluster
* mila cluster

For example, to run on the mila cluster with orion:

    cd examples/slurm_mila_orion/exp01
    sbatch run.sh

This example will run orion for just one trial (see the orion config file).
Note the folder structure. The root folder for this example is
`examples/slurm_mila_orion`.
Here you can find the orion config file (`orion_config.yaml`), as well as the config
file for your project (that contains the hyper-parametersi - `config.yaml`).
Here the code will write the orion db file and the mlruns folder
(i.e., the folder containing the mlflow results).

The `exp01` folder contains the executable (`run.sh`), and after running
it will contain the log files (in `logs`)as well as the experiment saved files
(e.g., the saved models) - note that the exp folder will contain a copy of the 
log as well (to keep all the files related to one trial into one folder).


The reason why there is a folder `exp01` is because you may want to run more
trials in parallel. To do so, just copy `exp01`, e.g.,

    cp -r exp01 exp02

and launch `run.sh` in both the folder. Orion will take care to sync the two
experiments.

[![Build Status](https://travis-ci.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}.png?branch=master)](https://travis-ci.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }})
[![codecov](https://codecov.io/gh/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/branch/master/graph/badge.svg)](https://codecov.io/gh/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }})

{% set is_open_source = cookiecutter.open_source_license != 'Not open source' -%}

# {{ cookiecutter.project_name }}


{{ cookiecutter.project_short_description }}

{% if is_open_source %}
* Free software: {{ cookiecutter.open_source_license }}
{% endif %}


## Instructions to setup the project

### Add git:

    git init

### Setup pre-commit hooks:
(this will run flake8 before any commit)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### Commit the code

    git add .
    git commit -m 'first commit'

### Link github to your local repository
Go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project, which should look like this:

    git remote add origin git@github.com:{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}.git
    git push -u origin master

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

    pip install -e .
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}

Note: if running tensorflow, you may need:

    pip install -U setuptools
{%- endif %}

### Add Travis
Follow the instructions here: https://docs.travis-ci.com/user/tutorial/
Note that the `.travis.yml` file is already in your repository (so, no need to
create it).
After giving permission to travis to access you repository, you should be able to
trigger a build (top-right - search for `More Options` => `Trigger Build`.

### Add Codecov
Go to https://codecov.io/ and enable codecov for your repository.
In particular, get the secret token for your project and add it to
github (see https://docs.codecov.io/docs/about-the-codecov-bash-uploader#section-upload-token).

### Run the tests
Just run (from the root folder):

    pytest

### Run the code
Note that the code should already compile at this point.
Try to run the examples under $ROOT/examples
(see next sections).

### Examples

The `examples` folder contains several execution examples.
There is a folder for running without orion and with orion.
Also, there is an example for the following environments:
* local (e.g., you laptop)
* mila cluster

For example, to run on the mila cluster with orion:

    cd examples/slurm_mila_orion/exp01
    sh run.sh

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

To setup pre-commit hooks (to use flake8 before avery commit) runs this (from the root folder):

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -


## YOUR PROJECT README:

* __TODO__


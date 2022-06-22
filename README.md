[![Build Status](https://api.travis-ci.com/mila-iqia/cookiecutter-pyml.svg?branch=master)](https://travis-ci.com/github/mila-iqia/cookiecutter-pyml)

About 
-----

A cookiecutter is a generic project template that will instantiate a new project with sane defaults. This repo contains our custom cookiecutter (`cookiecutter-pyml`) which will generate a new python deep learning package preconfigured with best practices in mind. It currently supports:

* Pytorch(PyTorch Lightning)
* Travis CI
* Sphinx (documentation)
* MLFlow (experiment management)
* Orion (hyperparameter optimization)
* Flake8
* Pytest

More information on what a cookiecutter is [here.](https://cookiecutter.readthedocs.io)

Quickstart
----------

Install the latest version of cookiecutter:

    pip install -U cookiecutter

Generate your project using our template. Make sure to use the command exactly as you see it here. 
This will use cookiecutter to instantiate your new project from our template (https://github.com/mila-iqia/cookiecutter-pyml.git).

    cookiecutter https://github.com/mila-iqia/cookiecutter-pyml.git

Follow the CLI instructions, then cd into your newly created project folder:

    cd $YOUR_PROJECT_NAME

Follow the instructions in the README in the newly created repository (`$YOUR_PROJECT_NAME/README.md`) to get started with your new project (in particular, the section "Instructions to setup the project").

Enjoy the cookies!

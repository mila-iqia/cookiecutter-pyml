__TODO_START__
When starting a project, the other company needs to approve the libraries/tools that we use.
There are two options that the company can choose:
- Send a generic approval (via email) that all the libraries that we decide to use are pre-approved.
- Keep a list of the libraries, and when a new library is added, add it to the list and ask the company to approve.

If the former is selected, you can delete this file.
If the latter, this file represents the list.

The way it works is the following:
- First, make sure that all the licenses listed below are still valid (i.e., they did not change). You can use the link to speedup this process.
- Note that the last time the list was checked was on 2023-April-04.
- Then, ask the company to approve the list by writing "approved" (or similar), with their git account (this will prove that the company approval is authentic).
- When new libraries are added, create a new section below (`Dependencies as of ...`), where you copy the old list and you modify it according to your needs.
- Ask the company to approve with the method above (git commit).

Those instructions between TODO_START and TODO_END can be deleted.

__TODO_END__

Here is the list of dependencies required by Mila for the project development.
Note that the following dependencies will trigger some other dependencies and for sake of simplicity,
we are not listing them all. Modifications since last approved version are in **bold**.

# Dependencies as of {PROJECT_START_DATE}

## General dependencies

* Python: [PSF](https://docs.python.org/3/license.html#psf-license) ([Python](https://www.python.org/))
* virtualenv : MIT ([https://pypi.org/project/virtualenv/](https://pypi.org/project/virtualenv/))
* miniconda : 3-clause BSD ([https://docs.conda.io/en/latest/license.html](https://docs.conda.io/en/latest/license.html))
* pip : MIT ([https://en.wikipedia.org/wiki/Pip_(package_manager)](https://en.wikipedia.org/wiki/Pip_(package_manager)))
* Mila cookiecutter : MIT [https://github.com/mila-iqia/cookiecutter-pyml/blob/development/LICENSE](https://github.com/mila-iqia/cookiecutter-pyml/blob/development/LICENSE)
* ZSH: MIT-like [zsh](http://zsh.sourceforge.net/)
* OH-MY-ZSH: MIT [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh/)
* TMUX - ISC [tmux](https://github.com/tmux/tmux/)
* [ipdb](https://pypi.org/project/ipdb/) BSD License (BSD), - A very useful debugger.
* [ipython](https://pypi.org/project/ipython/) BSD, - A very useful interactive ipython shell

## Library dependencies
(see `setup.py`)

* 'flake8', MIT ([https://pypi.org/project/flake8/](https://pypi.org/project/flake8/))
* 'flake8-docstrings', MIT ([https://pypi.org/project/flake8-docstrings/](https://pypi.org/project/flake8-docstrings/))
* 'gitpython', BSD ([https://pypi.org/project/GitPython/](https://pypi.org/project/GitPython/))
* 'jupyter', BSD ([https://github.com/jupyter/jupyter/blob/master/LICENSE](https://github.com/jupyter/jupyter/blob/master/LICENSE))
* 'jinja2', [3-clause BSD](https://jinja.palletsprojects.com/en/3.1.x/license/) ([https://palletsprojects.com/p/jinja/](https://palletsprojects.com/p/jinja/))
* 'myst-parser, MIT, ([https://github.com/executablebooks/MyST-Parser](https://github.com/executablebooks/MyST-Parser))
* 'orion', BSD [https://github.com/Epistimio/orion/blob/develop/LICENSE](https://github.com/Epistimio/orion/blob/develop/LICENSE)
* 'pyyaml', MIT ([https://pypi.org/project/PyYAML/](https://pypi.org/project/PyYAML/))
* 'pytest', MIT ([https://pypi.org/project/pytest/](https://pypi.org/project/pytest/))
* 'pytest-cov', MIT ([https://pypi.org/project/pytest-cov/](https://pypi.org/project/pytest-cov/))
* 'pytype', MIT + Apache 2.0, ([https://github.com/google/pytype/](https://github.com/google/pytype/blob/master/LICENSE))
* 'pytorch_lightning', Apache-2.0 ([https://pypi.org/project/pytorch-lightning/](https://pypi.org/project/pytorch-lightning/))
* 'sphinx', BSD ([https://pypi.org/project/Sphinx/](https://pypi.org/project/Sphinx/))
* 'sphinx-autoapi', MIT ([https://pypi.org/project/sphinx-autoapi/](https://pypi.org/project/sphinx-autoapi/))
* 'sphinx-rtd-theme', MIT ([https://pypi.org/project/sphinx-rtd-theme/](https://pypi.org/project/sphinx-rtd-theme/))
* 'sphinxcontrib-napoleon', BSD ([https://pypi.org/project/sphinxcontrib-napoleon/](https://pypi.org/project/sphinxcontrib-napoleon/))
* 'sphinxcontrib-katex', MIT ([https://pypi.org/project/sphinxcontrib-katex/](https://pypi.org/project/sphinxcontrib-katex/))
* 'tensorboard', Apache License 2.0 ([https://github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard))
* 'tqdm', MIT+Mozilla Public license ([https://pypi.org/project/tqdm/](https://pypi.org/project/tqdm/))
* 'torch', BSD-3 ([https://pypi.org/project/torch/](https://pypi.org/project/torch/))
* 'torchvision', BSD, ([https://pypi.org/project/torchvision/](https://pypi.org/project/torchvision/))

## Pre-trained models
Add any pre-trained models and associated licenses here if relevant.
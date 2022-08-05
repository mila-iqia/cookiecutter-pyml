{%- if cookiecutter.environment == 'mila' %}
Here is the list of dependencies required by Mila for the project development.
Note that the following dependencies will trigger some other dependencies and for sake of simplicity,
we are not listing them all. Modifications since last approved version are in **bold**.

# Dependencies as of {% now 'local', '%Y/%m/%d' %}

## General dependencies

* Python: [PSF](https://docs.python.org/3/license.html#psf-license) ([Python](https://www.python.org/))
* virtualenv : MIT ([https://pypi.org/project/virtualenv/](https://pypi.org/project/virtualenv/))
* miniconda : 3-clause BSD ([https://docs.conda.io/en/latest/license.html](https://docs.conda.io/en/latest/license.html))
* pip : MIT ([https://en.wikipedia.org/wiki/Pip_(package_manager)](https://en.wikipedia.org/wiki/Pip_(package_manager)))
* Mila cookiecutter : MIT [https://github.com/mila-iqia/cookiecutter-pyml/tree/lightning_and_keras](https://github.com/mila-iqia/cookiecutter-pyml/tree/lightning_and_keras)
* ZSH: MIT-like [zsh](http://zsh.sourceforge.net/)
* OH-MY-ZSH: MIT [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh/)
* TMUX - ISC [tmux](https://github.com/tmux/tmux/)

## Library dependencies
(see `setup.py`)

* 'flake8', MIT ([https://pypi.org/project/flake8/](https://pypi.org/project/flake8/))
* 'flake8-docstrings', MIT ([https://pypi.org/project/flake8-docstrings/](https://pypi.org/project/flake8-docstrings/))
* 'gitpython', BSD ([https://pypi.org/project/GitPython/](https://pypi.org/project/GitPython/))
* 'jupyter', BSD ([https://github.com/jupyter/jupyter/blob/master/LICENSE](https://github.com/jupyter/jupyter/blob/master/LICENSE))
* 'jupyter notebook', BSD ([https://github.com/jupyter/notebook/blob/master/LICENSE](https://github.com/jupyter/notebook/blob/master/LICENSE))
* 'jinja2', [3-clause BSD](https://jinja.palletsprojects.com/en/3.1.x/license/) ([https://palletsprojects.com/p/jinja/](https://palletsprojects.com/p/jinja/))
* 'orion', BSD [https://github.com/Epistimio/orion/blob/develop/LICENSE](https://github.com/Epistimio/orion/blob/develop/LICENSE)
* 'pyyaml', MIT ([https://pypi.org/project/PyYAML/](https://pypi.org/project/PyYAML/))
* 'pytest', MIT ([https://pypi.org/project/pytest/](https://pypi.org/project/pytest/))
* 'pytest-cov', BSD (MIT) ([https://pypi.org/project/pytest-cov/](https://pypi.org/project/pytest-cov/))
* 'pytype', MIT + Apache 2.0, ([https://github.com/google/pytype/](https://github.com/google/pytype/blob/master/LICENSE))
* 'pytorch_lightning', Apache-2.0 ([https://pypi.org/project/pytorch-lightning/](https://pypi.org/project/pytorch-lightning/))
* 'sphinx', BSD ([https://pypi.org/project/Sphinx/](https://pypi.org/project/Sphinx/))
* 'sphinx-autoapi', MIT ([https://pypi.org/project/sphinx-autoapi/](https://pypi.org/project/sphinx-autoapi/))
* 'sphinx-rtd-theme', MIT ([https://pypi.org/project/sphinx-rtd-theme/](https://pypi.org/project/sphinx-rtd-theme/))
* 'sphinxcontrib-napoleon', BSD ([https://pypi.org/project/sphinxcontrib-napoleon/](https://pypi.org/project/sphinxcontrib-napoleon/))
* 'sphinxcontrib-katex', BSD (MIT) ([https://pypi.org/project/sphinxcontrib-katex/](https://pypi.org/project/sphinxcontrib-katex/))
* 'recommonmark', MIT ([https://pypi.org/project/recommonmark/](https://pypi.org/project/recommonmark/))
* 'tensorboard', Apache License 2.0 ([https://github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard))
* 'tqdm', MIT ([https://pypi.org/project/tqdm/](https://pypi.org/project/tqdm/))
* 'torch', BSD-3 ([https://pypi.org/project/torch/](https://pypi.org/project/torch/))

## External dependencies to the project

* [ipdb](https://pypi.org/project/ipdb/) BSD License (BSD), - A very useful debugger.
* [ipython](https://pypi.org/project/ipython/) BSD, - A very useful interactive ipython shell
* [jupytext](https://jupytext.readthedocs.io/en/latest/index.html)  MIT, - this library allows us to easily convert and sync .ipynb and .py files simultaneously - very useful for notebooks without needing a browser.
* [pepermill](https://papermill.readthedocs.io/en/latest/) BSD, - very useful for running jupyter notebooks end-to-end directly from the CLI
* [jupyter-vim-binding](https://github.com/lambdalisue/jupyter-vim-binding), MIT

## Pre-trained models

* ADD IF USED 
{%- endif %}
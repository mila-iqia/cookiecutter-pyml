# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}

cookiecutter ../.. --no-input --output-dir=./
cd wonderful_project
pip install -e .'[dev]' --quiet

# necessary cause tf dependencies are sometimes not updated
pip install -U setuptools numpy six --quiet

# Build the docs
cd docs
sphinx-build -b html -d _build/doctrees . _build/html

# exit at the first error
set -e

cookiecutter ../.. --no-input --output-dir=./
cd wonderful_project
pip install -e . --quiet

# necessary cause tf dependencies are sometimes not updated
pip install -U setuptools numpy six --quiet

# Build the docs
cd docs
sphinx-build -b html -d _build/doctrees . _build/html

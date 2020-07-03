# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}

cookiecutter ../.. --no-input --output-dir=./
cd wonderful_project
pip install -e . --quiet

#run flake8 test first
pip install flake8 --quiet
sh config/hooks/pre-commit

# run the example
cd examples/local
sh run.sh
cd ../..
cd examples/local_orion
sh run.sh

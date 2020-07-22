# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}

python -c "from cookiecutter.main import cookiecutter; cookiecutter('../..', no_input=True, extra_context={'dl_framework': 'tensorflow_cpu'}, output_dir='./')"
cd wonderful_project
pip install -e . --quiet
pip install flake8 pytest --quiet
# necessary cause tf dependencies are sometimes not updated
pip install -U setuptools numpy six --quiet

# print all dependencies
pip freeze

# run flake8 test first
sh config/hooks/pre-commit

# run tests
pytest .

# run the examples
cd examples/local
sh run.sh
cd ../../
cd examples/local_orion
sh run.sh

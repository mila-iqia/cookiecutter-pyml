# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}

python -c "from cookiecutter.main import cookiecutter; cookiecutter('../..', no_input=True, extra_context={'dl_framework': 'tensorflow_cpu'}, output_dir='./')"
cd wonderful_project
pip install -e . --quiet
# necessary cause tf dependencies are sometimes not updated
pip install -U setuptools numpy six --quiet

#run flake8 test first
pip install flake8 --quiet
sh config/hooks/pre-commit

# run the examples
cd examples/local
sh run.sh
cd ../../
cd examples/local_orion
sh run.sh

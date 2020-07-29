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
mv output outout_OLD
# re-run the example to check reproducibility
sh run.sh
# check results are the same
DIFF_LINES=`grep "best_dev_metric" output*/stats.yaml | sed 's@^.*best_dev_metric: @@g' | uniq | wc -l`
if [ ${DIFF_LINES} -gt 1 ]; then
    echo "ERROR: two identical runs produced different output results - review seed implementation"
    exit 1
else
    echo "PASS: two identical runs produced the same output results."
fi

# run Orion
cd ../..
cd examples/local_orion
sh run.sh
mv orion_working_dir orion_working_dir_OLD
# re-run the example to check reproducibility
rm -fr orion_db*
sh run.sh
DIFF_LINES=`grep "best_dev_metric" orion_working_dir*/*/stats.yaml | sed 's@^.*best_dev_metric: @@g' | sort | uniq | wc -l`
if [ ${DIFF_LINES} -gt 2 ]; then
    echo "ERROR: two identical Orion runs produced different output results - review seed implementation"
    exit 1
else
    echo "PASS: two identical Orion runs produced the same output results."
fi

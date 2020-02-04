python -c "from cookiecutter.main import cookiecutter; cookiecutter('../..', no_input=True, extra_context={'dl_framework': 'tensorflow_cpu'}, output_dir='./')"
cd wonderful_project
pip install -e . --quiet
pip install -U setuptools numpy six --quiet
cd examples/local
pwd
main --data data --output output --config config.yaml --disable_progressbar

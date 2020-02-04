cookiecutter ../.. --no-input --output-dir=./
cd wonderful_project
pip install -e . --quiet
cd examples/local
pwd
main --data data --output output --config config.yaml --disable_progressbar

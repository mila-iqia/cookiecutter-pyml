# exit at the first error
set -e

# Build the docs
cd ./docs/
sphinx-build -b html -d _build/doctrees . _build/html

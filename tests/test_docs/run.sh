# exit at the first error
set -e

# Build the docs
cd $GITHUB_WORKSPACE/docs/
sphinx-build -b html -d _build/doctrees . _build/html

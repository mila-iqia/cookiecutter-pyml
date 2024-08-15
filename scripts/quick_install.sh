# This script is used to quickly install the cookiecutter-pyml template
set -e

# Default value for project_name
project_name="amlrt_project"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project-name) project_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use the project_name variable
echo "Cloning cookie-cutter to: $project_name..."

# TODO: Before merging, revert back to installing from main branch
# git clone https://github.com/mila-iqia/cookiecutter-pyml.git $project_name
git clone --branch development https://github.com/mila-iqia/cookiecutter-pyml.git $project_name

# Remove the .git folder and reinitialize it
echo "Initializing the git repository..."
cd $project_name
rm -fr .git
git init

# Replace the README.md file
mv scripts/README.new.md README.md

echo "Done! You can now visit your project by navigating to it:"
echo "cd $project_name"

echo "Remember to point it to your github repository:"
echo "git remote add origin git@github.com:\${GITHUB_USERNAME}/\${PROJECT_NAME}.git"
echo ""
echo "For more information, please visit https://github.com/mila-iqia/cookiecutter-pyml"

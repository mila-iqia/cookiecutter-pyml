# This script is used to quickly install the cookiecutter-pyml template
set -e

# Default value for project_name
project_name="amlrt_project"


replace_project_name() {
    # Replace all instances of amlrt_project with the project name and rename the root folder
    local project_name="$1"

    # Check if the OS is macOS or Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS

        # Replace all instances of amlrt_project with the project name
        find . -type f -exec grep -l 'amlrt_project' {} \; | xargs sed -i '' 's/amlrt_project/'"$project_name"'/g'

        # Rename root folder
        mv amlrt_project "$project_name"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux

        # Replace all instances of amlrt_project with the project name
        find . -type f -exec grep -l 'amlrt_project' {} \; | xargs sed -i 's/amlrt_project/'"$project_name"'/g'

        # Rename root folder
        mv amlrt_project "$project_name"
    else
        echo "Unsupported OS: $OSTYPE"
        echo "Your OS is not yet supported. You will have to manually replace all instances of amlrt_project with your project name."
    fi
}


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

# TODO: Once this has been properly merged to master, update ref to point to master
# git clone https://github.com/mila-iqia/cookiecutter-pyml.git $project_name
git clone --branch development https://github.com/mila-iqia/cookiecutter-pyml.git $project_name

# Remove the .git folder and reinitialize it
echo "Initializing the git repository..."
cd $project_name
rm -fr .git
git init

replace_project_name $project_name

# Replace the README.md file
mv scripts/README.new.md README.md

echo ""
echo "Done! You can now visit your project by navigating to it:"
echo ""
echo "   cd $project_name"

echo ""
echo "Remember to point it to your github repository:"
echo "git remote add origin git@github.com:\${GITHUB_USERNAME}/\${PROJECT_NAME}.git"
echo ""
echo "For more information, please visit https://github.com/mila-iqia/cookiecutter-pyml"

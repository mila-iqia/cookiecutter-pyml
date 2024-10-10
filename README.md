# AMLRT Cookiecutter - Initialize a new project


## Automatic Install (recommended)

For convenience, you can run:

    bash <(curl -s https://raw.githubusercontent.com/mila-iqia/cookiecutter-pyml/master/scripts/quick_install.sh) --project-name my_new_project

replace `my_new_project` with the name of your project (the default value is `amlrt_project`). This will clone and setup the cookiecutter for you in the newly created folder `my_new_project`.

Once done, go to the [First Commit](#first-commit) section and follow the instructions.

Note: if the `my_new_project` folder already exists, the installation will not proceed.

## Manual Install

Note: Skip to next section if you used the automatic install

First, git clone this project template locally.

    git clone https://github.com/mila-iqia/cookiecutter-pyml.git

Select a name for the new project; in the following we assume that
the name is `${PROJECT_NAME}`. Change it accordingly to the correct name.

Rename your just-cloned folder to the new project name:

    mv cookiecutter-pyml ${PROJECT_NAME}

Now go into the project folder and delete the git history.

    cd ${PROJECT_NAME}
    rm -fr .git

This is done so that your new project will start with a clean git history.
Now, initialize the repository with git:

    git init

You can now replace this README.md file with the standard README file for a project.
    mv scripts/README.md.example README.md

# First Commit

To perform your first commit:

    git add .
    git commit -m 'first commit'

Next, go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project, which should look like this:
(PS: these instructions are reported here for your convenience.
We suggest to also look at the GitHub project page for more up-to-date info)

    git remote add origin git@github.com:${GITHUB_USERNAME}/${PROJECT_NAME}.git
    git branch -M main
    git push -u origin main

At this point, the local code is versioned with git and pushed to GitHub.
You will not need to use the instructions in this section anymore, so we
suggest to delete this section ("AMLRT Cookiecutter - Initialize a new project") entirely.
(by doing so it will be clear that the initialization has been already done,
and all you need from now on is just to git clone from the repository you
just pushed, i.e., `git@github.com:${GITHUB_USERNAME}/${PROJECT_NAME}.git`).

Once you have successfully completed these steps, you can remove this README.md and update it with the template README provided. Adapt it to your needs:

    mv scripts/README.new.md README.md

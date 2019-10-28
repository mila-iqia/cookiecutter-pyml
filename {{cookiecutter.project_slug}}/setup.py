from setuptools import setup, find_packages


setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
    install_requires=[
        'tqdm', 'torch', 'mlflow', 'orion', 'pyyaml', 'torchtext', 'nltk', 'spacy'],
    entry_points={
        'console_scripts': [
            'main={{ cookiecutter.project_slug }}.main:main'
        ],
    }
)

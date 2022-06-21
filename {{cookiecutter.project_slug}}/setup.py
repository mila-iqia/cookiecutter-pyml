from setuptools import setup, find_packages


setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
    python_requires='>={{ cookiecutter.python_version }}',
    install_requires=[
        'flake8',
        'flake8-docstrings',
        'gitpython',
        'tqdm',
        'jupyter',
        'mlflow==1.15.0',
        'orion>=0.1.14',
        'pyyaml>=5.3',
        'pytest>=4.6',
        'pytest-cov',
        'sphinx==3.5.4',
        'sphinx-autoapi==1.8.1',
        'sphinx-rtd-theme==0.5.2',
        'sphinxcontrib-napoleon==0.7',
        'sphinxcontrib-katex==0.7.2',
        'recommonmark==0.7.1',
        'torch==1.8.1',
        'pytorch_lightning==1.2.7'],
    entry_points={
        'console_scripts': [
            'main={{ cookiecutter.project_slug }}.main:main'
        ],
    }
)

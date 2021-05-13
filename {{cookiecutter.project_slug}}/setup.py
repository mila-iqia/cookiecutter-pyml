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
        'mlflow==1.10.0',
        'orion>=0.1.8',
        'pyyaml>=5.3',
        'pytest>=4.6',
        'sphinx',
        'sphinx-autoapi',
        'sphinx-rtd-theme',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-katex',
        'recommonmark',
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        'torch', 'pytorch_lightning==1.0.6'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_cpu' %}
        'tensorflow'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_gpu' %}
        'tensorflow-gpu'],
        {%- endif %}
    entry_points={
        'console_scripts': [
            'main={{ cookiecutter.project_slug }}.main:main'
        ],
    }
)

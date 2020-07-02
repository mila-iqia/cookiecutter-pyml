from setuptools import setup, find_packages


setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
    python_requires='>={{ cookiecutter.python_version }}',
    install_requires=[
        'flake8', 'tqdm', 'mlflow', 'orion', 'pyyaml>=5.3', 'pytest',
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        'torch'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_cpu' %}
        'scipy==1.4.1', 'tensorflow==2.2.0', 'setuptools>=41.0.0'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_gpu' %}
        'scipy==1.4.1', 'tensorflow-gpu==2.2.0', 'setuptools>=41.0.0'],
        {%- endif %}
    entry_points={
        'console_scripts': [
            'main={{ cookiecutter.project_slug }}.main:main'
        ],
    }
)

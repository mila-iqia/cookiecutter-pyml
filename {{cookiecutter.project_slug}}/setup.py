from setuptools import setup, find_packages


setup(
    name='{{ cookiecutter.project_slug }}',
    version='{{ cookiecutter.version }}',
    packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
    python_requires='>={{ cookiecutter.python_version }}',
    install_requires=[
        'tqdm', 'mlflow', 'orion', 'pyyaml',
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        'torch'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_cpu' %}
        'tensorflow>=2.0'],
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'tensorflow_gpu' %}
        'tensorflow-gpu>=2.0'],
        {%- endif %}
    entry_points={
        'console_scripts': [
            'main={{ cookiecutter.project_slug }}.main:main'
        ],
    }
)

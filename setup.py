from setuptools import find_packages, setup

setup(
    name='amlrt_project',
    version='0.0.1',
    packages=find_packages(include=['amlrt_project', 'amlrt_project.*']),
    python_requires='>=3.9',
    install_requires=[
        'flake8==4.0.1',
        'flake8-docstrings==1.6.0',
        'gitpython==3.1.27',
        'jupyter==1.0.0',
        'jinja2==3.1.2',
        'myst-parser==2.0.0',
        'orion>=0.2.4.post1',
        'pyyaml==6.0',
        'pytest==7.1.2',
        'pytest-cov==3.0.0',
        'pytorch_lightning==1.8.3',
        'pytype==2024.2.27',
        'sphinx==7.2.6',
        'sphinx-autoapi==3.0.0',
        'sphinx-rtd-theme==1.3.0',
        'sphinxcontrib-napoleon==0.7',
        'sphinxcontrib-katex==0.9.9',
        'tensorboard==2.16.2',
        'tqdm==4.64.0',
        'torch==2.2.1',
        'torchvision==0.17.1',
        'torchmetrics==0.11.0'
    ],
    entry_points={
        'console_scripts': [
            # TODO: change amlrt- prefix, placeholder for now.
            'amlrt-train=amlrt_project.train:main',
            'amlrt-eval=amlrt_project.evaluate:main',
        ],
    }
)

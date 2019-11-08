export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config orion_config.yaml ../../{{cookiecutter.project_slug}}/main.py --data no_data --output no_output --config config.yaml

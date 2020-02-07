export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config orion_config.yaml ../../{{cookiecutter.project_slug}}/main.py --data data \
    --config config.yaml --disable_progressbar \
    --output '{exp.working_dir}/{exp.name}_{trial.id}/' \
    --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'
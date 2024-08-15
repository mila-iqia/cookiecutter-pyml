set -e
export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

amlrt_project_merge_configs --config ../config.yaml config.yaml --merged-config-file merged_config.yaml
orion -vvv -v hunt --config orion_config.yaml amlrt_project_train --data ../data \
    --config merged_config.yaml --disable-progressbar \
    --output '{exp.working_dir}/{trial.id}/' \
    --log '{exp.working_dir}/{trial.id}/exp.log'

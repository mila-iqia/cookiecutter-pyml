import os

import yaml

if os.name != 'nt':
    # not using AIM on Windows
    from aim.pytorch_lightning import AimLogger

from amlrt_project.data.constants import LOG_FOLDER

AIM_INFO_FILE_NAME = "aim_info.yaml"


def prepare_aim_logger(hyper_params, options, output):
    """Create the aim logger - make sure to track on the same experiment if resuming one."""
    if LOG_FOLDER not in options:
        raise ValueError('please set log_folder in config file to use aim')
    aim_run_info_dict = retrieve_aim_run_info(
        output, hyper_params["exp_name"], options[LOG_FOLDER],
    )
    aim_logger = AimLogger(
        run_name=aim_run_info_dict["run_name"] if aim_run_info_dict else None,
        run_hash=aim_run_info_dict["run_hash"] if aim_run_info_dict else None,
        experiment=hyper_params["exp_name"],
        repo=options[LOG_FOLDER],
        train_metric_prefix="train__",
        val_metric_prefix="val__",
    )
    # get orion trail id if using orion - if yes, this will be used as the run name
    orion_trial_id = os.environ.get("ORION_TRIAL_ID")
    if orion_trial_id:
        aim_logger.experiment.name = orion_trial_id
    save_aim_run_info(
        aim_logger.experiment.name,
        aim_logger.experiment.hash,
        output,
        hyper_params["exp_name"],
        options[LOG_FOLDER],
    )
    return aim_logger


def save_aim_run_info(
    run_name: str,
    run_hash: str,
    output: str,
    experiment: str,
    repo: str,
):
    """Save aim_run_info_dict to output dir."""
    aim_run_info_dict = {
        "experiment": experiment,
        "aim_dir": repo,
        "run_name": run_name,
        "run_hash": run_hash,
    }
    with open(os.path.join(output, AIM_INFO_FILE_NAME), "w") as file:
        yaml.dump(aim_run_info_dict, file)


def retrieve_aim_run_info(
    output: str,
    experiment: str,
    repo: str,
):
    """Retrieve aim_run_info_dict from previous run's output dir."""
    if os.path.exists(os.path.join(output, AIM_INFO_FILE_NAME)):
        # output exist and aim_info.yaml exists under output
        # this means current run is not starting from scratch
        # so we will try to load aim_info.yaml to resume the previous run
        with open(os.path.join(output, AIM_INFO_FILE_NAME), "r") as file:
            aim_run_info_dict = yaml.load(file, Loader=yaml.FullLoader)
        if (experiment != aim_run_info_dict["experiment"]) or (
            repo != aim_run_info_dict["aim_dir"]
        ):
            # if the experiment changes or the aim logging directory changes
            # either of these means the run is differently
            # so will not resume the previous run for aim
            aim_run_info_dict = None
    else:
        aim_run_info_dict = None

    return aim_run_info_dict

import logging
import os
import socket

import pytorch_lightning as pl
import yaml

if os.name != 'nt':
    # not using AIM on Windows
    from aim.pytorch_lightning import AimLogger

from git import InvalidGitRepositoryError, Repo
from pip._internal.operations import freeze

from amlrt_project.data.constants import (AIM, EXP_LOGGERS, LOG_FOLDER,
                                          TENSORBOARD)

logger = logging.getLogger(__name__)
AIM_INFO_FILE_NAME = "aim_info.yaml"


class LoggerWriter:  # pragma: no cover
    """LoggerWriter.

    see: https://stackoverflow.com/questions/19425736/
    how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, printer):
        """__init__.

        Args:
            printer: (fn) function used to print message (e.g., logger.info).
        """
        self.printer = printer
        self.encoding = None

    def write(self, message):
        """write.

        Args:
            message: (str) message to print.
        """
        if message != '\n':
            self.printer(message)

    def flush(self):
        """flush."""
        pass


def get_git_hash(script_location):  # pragma: no cover
    """Find the git hash for the running repository.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :return: (str) the git hash for the repository of the provided script.
    """
    if not script_location.endswith('.py'):
        raise ValueError('script_location should point to a python script')
    repo_folder = os.path.dirname(script_location)
    try:
        repo = Repo(repo_folder, search_parent_directories=True)
        commit_hash = repo.head.commit
    except (InvalidGitRepositoryError, ValueError):
        commit_hash = 'git repository not found'
    return commit_hash


def log_exp_details(script_location, args):  # pragma: no cover
    """Will log the experiment details to both screen logger and mlflow.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :param args: the argparser object.
    """
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    dependencies = freeze.freeze()
    details = "\nhostname: {}\ngit code hash: {}\ndata folder: {}\ndata folder (abs): {}\n\n" \
              "dependencies:\n{}".format(
                  hostname, git_hash, args.data, os.path.abspath(args.data),
                  '\n'.join(dependencies))
    logger.info('Experiment info:' + details + '\n')


def load_experiment_loggers(
        hyper_params: dict,
        output: str):
    """Prepares and loads the loggers for this experiment.

    :param hyper_params: the experiment hyper-parameters
    :param output: the output folder
    :return: a dict containing the name and the associated logger
    """
    name2loggers = {}
    for logger_name, options in hyper_params[EXP_LOGGERS].items():
        if logger_name == TENSORBOARD:
            tb_logger = pl.loggers.TensorBoardLogger(
                save_dir=output,
                default_hp_metric=False,
                version=0,  # Necessary to resume tensorboard logging
            )
            name2loggers[TENSORBOARD] = tb_logger
        elif logger_name == AIM:
            if os.name == 'nt':
                logger.warning("AIM logger is not supported on Windows, skipped")
                continue
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
            name2loggers[AIM] = aim_logger
        else:
            raise NotImplementedError(f"logger {logger_name} is not supported")
    return name2loggers


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


def log_hyper_parameters(name2loggers, hyper_params, best_dev_result=None):
    """Log the experiment hyper-parameters to all the loggers."""
    for name, logger in name2loggers.items():
        if name == AIM:
            logger.log_hyperparams(hyper_params)
        elif name == TENSORBOARD:
            if best_dev_result is not None:
                logger.log_hyperparams(hyper_params, metrics={'best_dev_metric': best_dev_result})

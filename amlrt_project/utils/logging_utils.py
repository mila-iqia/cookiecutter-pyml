import logging
import os
import socket

import pytorch_lightning as pl
from git import InvalidGitRepositoryError, Repo
from pip._internal.operations import freeze
from pytorch_lightning.loggers import CometLogger

from amlrt_project.data.constants import AIM, COMET, EXP_LOGGERS, TENSORBOARD
from amlrt_project.utils.aim_logger_utils import prepare_aim_logger

logger = logging.getLogger(__name__)


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
            aim_logger = prepare_aim_logger(hyper_params, options, output)
            name2loggers[AIM] = aim_logger
        elif logger_name == COMET:
            comet_logger = CometLogger()
            name2loggers[COMET] = comet_logger
        else:
            raise NotImplementedError(f"logger {logger_name} is not supported")
    return name2loggers


def log_hyper_parameters(name2loggers, hyper_params):
    """Log the experiment hyper-parameters to all the loggers."""
    for name, logger in name2loggers.items():
        logger.log_hyperparams(hyper_params)

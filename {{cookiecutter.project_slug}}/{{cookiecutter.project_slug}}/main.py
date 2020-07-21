#!/usr/bin/env python

import argparse
import logging
import os
import sys

import mlflow
import yaml
from yaml import load

from {{cookiecutter.project_slug}}.data.data_loader import load_data
from {{cookiecutter.project_slug}}.train import train, load_stats, STAT_FILE_NAME
from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp
from {{cookiecutter.project_slug}}.models.model_loader import load_model
from {{cookiecutter.project_slug}}.models.model_loader import load_optimizer
from {{cookiecutter.project_slug}}.models.model_loader import load_loss
from {{cookiecutter.project_slug}}.utils.logging_utils import LoggerWriter, log_exp_details

logger = logging.getLogger(__name__)


def main():
    """Main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run.sh` file to see how to use it.

    """
    parser = argparse.ArgumentParser()
    # __TODO__ check you need all the following CLI parameters
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format')
    parser.add_argument('--data', help='path to data', required=True)
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will not load any existing saved model - even if present')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    # to intercept any print statement:
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.warning)

    if args.config is not None:
        with open(args.config, 'r') as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
    else:
        hyper_params = {}

    # to be done as soon as possible otherwise mlflow will not log with the proper exp. name
    if 'exp_name' in hyper_params:
        mlflow.set_experiment(hyper_params['exp_name'])
    if os.path.exists(os.path.join(args.output, STAT_FILE_NAME)):
        _, _, _, mlflow_run_id = load_stats(args.output)
        mlflow.start_run(run_id=mlflow_run_id)
    else:
        mlflow.start_run()
    run(args, hyper_params)
    mlflow.end_run()


def run(args, hyper_params):
    """Setup and run the dataloaders, training loops, etc.

    Args:
        args (list): arguments passed from the cli
        hyper_params (dict): hyper parameters from the config file
    """
    log_exp_details(os.path.realpath(__file__), args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    check_and_log_hp(
        ['batch_size', 'optimizer', 'patience', 'architecture', 'max_epoch',
         'exp_name'],
        hyper_params)

    train_loader, dev_loader = load_data(args, hyper_params)
    model = load_model(hyper_params)
    optimizer = load_optimizer(hyper_params, model)
    loss_fun = load_loss(hyper_params)

    train(model, optimizer, loss_fun, train_loader, dev_loader, hyper_params['patience'],
          args.output, max_epoch=hyper_params['max_epoch'],
          use_progress_bar=not args.disable_progressbar, start_from_scratch=args.start_from_scratch)


if __name__ == '__main__':
    main()

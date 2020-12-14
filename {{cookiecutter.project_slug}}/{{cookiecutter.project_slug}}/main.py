#!/usr/bin/env python

import argparse
import logging
import os
import sys

{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import mlflow
{%- endif %}
import yaml
from yaml import load

{%- if cookiecutter.dl_framework == 'pytorch' %}
from pytorch_lightning.loggers import MLFlowLogger
{%- endif %}
from {{cookiecutter.project_slug}}.data.data_loader import load_data
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
from {{cookiecutter.project_slug}}.train import train, load_stats, STAT_FILE_NAME
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
from {{cookiecutter.project_slug}}.train import train
{%- endif %}
from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp
from {{cookiecutter.project_slug}}.models.model_loader import load_model
from {{cookiecutter.project_slug}}.models.model_loader import load_optimizer
from {{cookiecutter.project_slug}}.models.model_loader import load_loss
from {{cookiecutter.project_slug}}.utils.file_utils import rsync_folder
from {{cookiecutter.project_slug}}.utils.logging_utils import LoggerWriter, log_exp_details
from {{cookiecutter.project_slug}}.utils.reproducibility_utils import set_seed

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
    parser.add_argument('--tmp-folder',
                        help='will use this folder as working folder - it will copy the input data '
                             'here, generate results here, and then copy them back to the output '
                             'folder')
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will not load any existing saved model - even if present')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.tmp_folder is not None:
        data_folder_name = os.path.basename(os.path.normpath(args.data))
        rsync_folder(args.data, args.tmp_folder)
        data_dir = os.path.join(args.tmp_folder, data_folder_name)
        output_dir = os.path.join(args.tmp_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        data_dir = args.data
        output_dir = args.output

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
    {%- if cookiecutter.dl_framework == 'pytorch' %}
    mlf_logger = MLFlowLogger(
        experiment_name=hyper_params.get('exp_name', 'Default')
    )
    run(args, data_dir, output_dir, hyper_params, mlf_logger)
    {%- endif %}
    {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
    if 'exp_name' in 'exp_name':
        mlflow.set_experiment(hyper_params['exp_name'])
    if os.path.exists(os.path.join(args.output, STAT_FILE_NAME)):
        _, _, _, mlflow_run_id = load_stats(args.output)
        mlflow.start_run(run_id=mlflow_run_id)
    else:
        mlflow.start_run()
    run(args, data_dir, output_dir, hyper_params)
    mlflow.end_run()
    {%- endif %}

    if args.tmp_folder is not None:
        rsync_folder(output_dir + os.path.sep, args.output)
{%- if cookiecutter.dl_framework == 'pytorch' %}


def run(args, data_dir, output_dir, hyper_params, mlf_logger):
{%- endif %}
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}


def run(args, data_dir, output_dir, hyper_params):
{%- endif %}
    """Setup and run the dataloaders, training loops, etc.

    Args:
        args (list): arguments passed from the cli
        data_dir (str): path to input folder
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
{%- if cookiecutter.dl_framework == 'pytorch' %}
        mlf_logger (obj): MLFlow logger callback.
{%- endif %}
    """
    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    logger.info('List of hyper-parameters:')
    check_and_log_hp(
        ['architecture', 'batch_size', 'exp_name', 'max_epoch', 'optimizer', 'patience', 'seed'],
        hyper_params)

    if hyper_params["seed"] is not None:
        set_seed(hyper_params["seed"])

    log_exp_details(os.path.realpath(__file__), args)

    train_loader, dev_loader = load_data(data_dir, hyper_params)
    model = load_model(hyper_params)
    optimizer = load_optimizer(hyper_params, model)
    loss_fun = load_loss(hyper_params)

    train(model=model, optimizer=optimizer, loss_fun=loss_fun, train_loader=train_loader,
          dev_loader=dev_loader, patience=hyper_params['patience'], output=output_dir,
          max_epoch=hyper_params['max_epoch'], use_progress_bar=not args.disable_progressbar,
{%- if cookiecutter.dl_framework == 'pytorch' %}
          start_from_scratch=args.start_from_scratch, mlf_logger=mlf_logger)
{%- endif %}
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
          start_from_scratch=args.start_from_scratch)
{%- endif %}


if __name__ == '__main__':
    main()

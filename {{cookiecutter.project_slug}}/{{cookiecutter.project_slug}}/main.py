#!/usr/bin/env python

import argparse
import logging
import os
import sys

import yaml
from yaml import load

from {{cookiecutter.project_slug}}.data.data_loader import load_data
from {{cookiecutter.project_slug}}.train import train
from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp
from {{cookiecutter.project_slug}}.models.model_loader import load_model
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
    parser.add_argument('--gpus', default=None,
                        help='list of GPUs to use. If not specified, runs on CPU.'
                             'Example of GPU usage: 1 means run on GPU 1, 0 on GPU 0.')
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

    run(args, data_dir, output_dir, hyper_params)

    if args.tmp_folder is not None:
        rsync_folder(output_dir + os.path.sep, args.output)


def run(args, data_dir, output_dir, hyper_params):
    """Setup and run the dataloaders, training loops, etc.

    Args:
        args (object): arguments passed from the cli
        data_dir (str): path to input folder
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
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

    datamodule = load_data(data_dir, hyper_params)
    model = load_model(hyper_params)

    train(model=model, datamodule=datamodule, output=output_dir, hyper_params=hyper_params,
          use_progress_bar=not args.disable_progressbar, start_from_scratch=args.start_from_scratch,
          gpus=args.gpus)


if __name__ == '__main__':
    main()

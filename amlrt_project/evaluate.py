import argparse
import logging
import sys

import comet_ml  # noqa
import pytorch_lightning as pl

from amlrt_project.data.data_loader import FashionMnistDM
from amlrt_project.models.model_loader import load_model
from amlrt_project.utils.config_utils import (
    add_config_file_params_to_argparser, load_configs)
from amlrt_project.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def main():
    """Main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run.sh` file to see how to use it.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument('--ckpt-path', help='Path to best model')
    parser.add_argument('--data', help='path to data', required=True)
    parser.add_argument('--accelerator', default='auto',
                        help='The accelerator to use - default is "auto".')
    add_config_file_params_to_argparser(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_dir = args.data

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    hyper_params = load_configs(args.config, args.cli_config_params)

    evaluate(args, data_dir, hyper_params)


def evaluate(args, data_dir, hyper_params):
    """Performs an evaluation on both the validation and test sets.

    Args:
        args (object): arguments passed from the cli
        data_dir (str): path to input folder
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
    """
    logger.info('List of hyper-parameters:')
    check_and_log_hp(
        ['architecture', 'batch_size', 'exp_name', 'early_stopping'],
        hyper_params)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
    )

    datamodule = FashionMnistDM(data_dir, hyper_params)
    datamodule.setup()

    model = load_model(hyper_params)

    val_metrics = trainer.validate(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

    logger.info(f"Validation Metrics: {val_metrics}")
    logger.info(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    main()

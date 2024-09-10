import argparse
import glob
import logging
import os
import shutil
import sys

import comet_ml  # noqa
import orion
import pytorch_lightning as pl
from orion.client import report_results
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from amlrt_project.data.data_loader import FashionMnistDM
from amlrt_project.models.model_loader import load_model
from amlrt_project.utils.config_utils import (
    add_config_file_params_to_argparser, load_configs, save_hparams)
from amlrt_project.utils.file_utils import rsync_folder
from amlrt_project.utils.hp_utils import check_and_log_hp
from amlrt_project.utils.logging_utils import (load_experiment_loggers,
                                               log_exp_details,
                                               log_hyper_parameters)
from amlrt_project.utils.reproducibility_utils import set_seed

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
CONFIG_FILE_NAME = 'config.yaml'


def main():
    """Main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run.sh` file to see how to use it.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument('--data', help='path to data', required=True)
    parser.add_argument('--tmp-folder',
                        help='will use this folder as working folder - it will copy the input data '
                             'here, generate results here, and then copy them back to the output '
                             'folder')
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will delete the output folder before starting the experiment')
    parser.add_argument('--accelerator', default='auto',
                        help='The accelerator to use - default is "auto".')
    parser.add_argument('--debug', action='store_true')
    add_config_file_params_to_argparser(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if os.path.exists(args.output) and args.start_from_scratch:
        logger.info('Starting from scratch, deleting the previous output folder.')
        shutil.rmtree(args.output)

    if os.path.exists(args.output):
        logger.info("Previous experiment found, resuming from checkpoint")
    else:
        os.makedirs(args.output)

    if args.tmp_folder is not None:
        data_folder_name = os.path.basename(os.path.normpath(args.data))
        rsync_folder(args.data, args.tmp_folder)
        data_dir = os.path.join(args.tmp_folder, data_folder_name)
        output_dir = os.path.join(args.tmp_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(args.output):
            rsync_folder(args.output, args.tmp_folder)
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

    hyper_params = load_configs(args.config, args.cli_config_params)
    save_hparams(hyper_params, os.path.join(args.output, CONFIG_FILE_NAME))

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
        ['architecture', 'batch_size', 'exp_name', 'max_epoch', 'optimizer', 'seed',
         'early_stopping'],
        hyper_params)

    if hyper_params["seed"] is not None:
        set_seed(hyper_params["seed"])

    log_exp_details(os.path.realpath(__file__), args)

    datamodule = FashionMnistDM(data_dir, hyper_params)
    model = load_model(hyper_params)

    train(model=model, datamodule=datamodule, output=output_dir, hyper_params=hyper_params,
          use_progress_bar=not args.disable_progressbar, accelerator=args.accelerator)


def train(**kwargs):  # pragma: no cover
    """Training loop wrapper. Used to catch exception if Orion is being used."""
    try:
        best_dev_metric = train_impl(**kwargs)
    except RuntimeError as err:
        if orion.client.cli.IS_ORION_ON and 'CUDA out of memory' in str(err):
            logger.error(err)
            logger.error('model was out of memory - assigning a bad score to tell Orion to avoid'
                         'too big model')
            best_dev_metric = -999
        else:
            raise err

    report_results([dict(
        name='dev_metric',
        type='objective',
        # note the minus - cause orion is always trying to minimize (cit. from the guide)
        value=-float(best_dev_metric))])


def train_impl(model, datamodule, output, hyper_params, use_progress_bar,
               accelerator):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        datamodule (obj): lightning data module that will instantiate data loaders.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        accelerator: the device where to run.
    """
    check_and_log_hp(['max_epoch'], hyper_params)

    best_model_path = os.path.join(output, BEST_MODEL_NAME)
    best_checkpoint_params = hyper_params['best_checkpoint']
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=best_model_path,
        filename='model',
        save_top_k=1,
        verbose=use_progress_bar,
        monitor=best_checkpoint_params['metric'],
        mode=best_checkpoint_params['mode'],
        every_n_epochs=best_checkpoint_params['every_n_epochs']
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=last_model_path,
        filename='model',
        verbose=use_progress_bar,
        every_n_epochs=1,
    )

    resume_from_checkpoint = handle_previous_models(output, last_model_path, best_model_path)

    early_stopping_params = hyper_params['early_stopping']
    check_and_log_hp(['metric', 'mode', 'patience'], hyper_params['early_stopping'])
    early_stopping = EarlyStopping(
        early_stopping_params['metric'],
        mode=early_stopping_params['mode'],
        patience=early_stopping_params['patience'],
        verbose=use_progress_bar)

    name2loggers = load_experiment_loggers(hyper_params, output)
    log_hyper_parameters(name2loggers, hyper_params)

    trainer = pl.Trainer(
        callbacks=[early_stopping, best_checkpoint_callback, last_checkpoint_callback],
        max_epochs=hyper_params['max_epoch'],
        accelerator=accelerator,
        logger=name2loggers.values()
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

    # Log the best result and associated hyper parameters
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    with open(os.path.join(output, 'results.txt'), 'w') as stream_out:
        stream_out.write(f'final best_dev_metric: {best_dev_result}\n')

    return best_dev_result


def handle_previous_models(output, last_model_path, best_model_path):
    """Moves the previous models in a new timestamp folder."""
    last_models = glob.glob(last_model_path + os.sep + '*')

    if len(last_models) >= 1:
        resume_from_checkpoint = sorted(last_models)[-1]
        logger.info(f'models found - resuming from {resume_from_checkpoint}')
    else:
        logger.info('no model found - starting training from scratch')
        resume_from_checkpoint = None
    return resume_from_checkpoint


if __name__ == '__main__':
    main()

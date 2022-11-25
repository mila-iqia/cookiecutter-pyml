import glob
import logging
import os

import orion
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_results

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp
from {{cookiecutter.project_slug}}.utils.callbacks import ComputeMetrics, LogConfusionMatrix

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'


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


def train_impl(model, datamodule, output, hyper_params, use_progress_bar, gpus):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        datamodule (obj): lightning data module that will instantiate data loaders.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        gpus: number of GPUs to use.
    """
    check_and_log_hp(['max_epoch'], hyper_params)

    best_model_path = os.path.join(output, BEST_MODEL_NAME)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=best_model_path,
        filename='model',
        save_top_k=1,
        verbose=use_progress_bar,
        monitor="val_loss",
        mode="max",
        every_n_epochs=1,
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=last_model_path,
        filename='model',
        verbose=use_progress_bar,
        every_n_epochs=1,
    )

    resume_from_checkpoint = handle_previous_models(output, last_model_path, best_model_path)
    compute_metrics = ComputeMetrics()
    log_conf_mats = LogConfusionMatrix()

    early_stopping_params = hyper_params['early_stopping']
    check_and_log_hp(['metric', 'mode', 'patience'], hyper_params['early_stopping'])
    early_stopping = EarlyStopping(
        early_stopping_params['metric'],
        mode=early_stopping_params['mode'],
        patience=early_stopping_params['patience'],
        verbose=use_progress_bar)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=output,
        default_hp_metric=False,
        version=0,  # Necessary to resume tensorboard logging
    )

    trainer = pl.Trainer(
        callbacks=[
            early_stopping,
            best_checkpoint_callback,
            last_checkpoint_callback,
            compute_metrics,
            log_conf_mats,
        ],
        max_epochs=hyper_params['max_epoch'],
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpus,
        logger=logger,
    )

    # run a full validation and report the values
    trainer.validate(model, datamodule=datamodule)

    # do the actual training
    trainer.fit(model, datamodule=datamodule)

    # Log the best result and associated hyper parameters
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    # NOTE: the hyper parameters only show up at the end of training.
    logger.log_hyperparams(hyper_params, metrics={'best_dev_metric': best_dev_result})

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

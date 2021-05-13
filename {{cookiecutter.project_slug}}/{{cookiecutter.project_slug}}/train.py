{%- if cookiecutter.dl_framework == 'pytorch' %}
import datetime
import glob
{%- endif %}
import logging
import os
{%- if cookiecutter.dl_framework == 'pytorch' %}
import shutil
{%- endif %}

import mlflow
import orion
import yaml
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import mlflow.tensorflow as mt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
{%- endif %}
from orion.client import report_results
from yaml import dump
from yaml import load

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
STAT_FILE_NAME = 'stats.yaml'


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
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}


def load_stats(output):
    """Load the latest statistics.

    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['best_dev_metric'], stats['epoch'], stats['remaining_patience'], \
        stats['mlflow_run_id']


def write_stats(output, best_eval_score, epoch, remaining_patience):
    """Write statistics of the best model at the end of every epoch.

    Args:
        output (str): Output directory
        best_eval_score (float): best score obtained on evaluation set.
        epoch (int): Which epoch training is at.
        remaining_patience (int): How many more epochs before training stops.
    """
    mlflow_run = mlflow.active_run()
    mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else 'NO_MLFLOW'
    to_store = {'best_dev_metric': best_eval_score, 'epoch': epoch,
                'remaining_patience': remaining_patience,
                'mlflow_run_id': mlflow_run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def reload_model(output, model_name, start_from_scratch=False):  # pragma: no cover
    """Reload a model.

    Can be useful for model checkpointing, hyper-parameter optimization, etc.

    Args:
        output (str): Output directory.
        model_name (str): Model name to relaod.
        start_from_scratch (bool): starts training from scratch even if a saved moel is present.
    """
    saved_model_folder = os.path.join(output, model_name)
    if start_from_scratch and os.path.exists(saved_model_folder):
        logger.info('saved model file "{}" already exists - but NOT loading it '
                    '(cause --start_from_scratch)'.format(output))
        restored = None
    elif os.path.exists(saved_model_folder):
        logger.info('loading model from {}'.format(saved_model_folder))
        model = tf.keras.models.load_model(
            os.path.join(output, LAST_MODEL_NAME)
        )

        stats = load_stats(output)
        logger.info('model status: {}'.format(stats))

        restored = model, stats
    else:
        logger.info('no model found to restore.')
        restored = None
    return restored


def init_model(model, train_loader):  # pragma: no cover
    """Init the model by computing a single forward pass on an input.

    Args:
        model (obj): The neural network model object.
        train_loader (obj): Dataloader for the training set.
    """
    model_input, model_target = next(iter(train_loader))
    _ = model(model_input)
    model.summary(print_fn=logger.info)


def train_impl(model, optimizer, loss_fun, train_loader, dev_loader, output, hyper_params,
               use_progress_bar=True, start_from_scratch=False):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        optimizer (obj): Optimizer used during training.
        loss_fun (obj): Loss function that will be optimized.
        train_loader (obj): Dataloader for the training set.
        dev_loader (obj): Dataloader for the validation set.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
    """
    check_and_log_hp(['max_epoch', 'patience'], hyper_params)

    restored = reload_model(output, LAST_MODEL_NAME, start_from_scratch)

    if restored is None:
        best_dev_metric = None
        remaining_patience = hyper_params['patience']
        start_epoch = 0

        model.compile(
            optimizer=optimizer,
            loss=loss_fun,
            metrics=[],
        )
    else:
        restored_model, stats = restored
        best_dev_metric, start_epoch, remaining_patience, _ = stats
        model = restored_model

    init_model(model, train_loader)

    es = EarlyStoppingWithModelSave(
        monitor='val_loss', min_delta=0, patience=hyper_params['patience'],
        verbose=use_progress_bar, mode='min', restore_best_weights=False, baseline=best_dev_metric,
        output=output, remaining_patience=remaining_patience
    )

    # set tensorflow/keras logging
    mt.autolog(every_n_iter=1)

    history = model.fit(x=train_loader, validation_data=dev_loader, callbacks=[es],
                        epochs=hyper_params['max_epoch'], verbose=use_progress_bar,
                        initial_epoch=start_epoch)

    best_val_loss = min(history.history['val_loss'])
    return best_val_loss


class EarlyStoppingWithModelSave(EarlyStopping):
    """Keras callback that extends the early stopping.

    Adds the functionality to save the models in the new TF format. (both best and last model)
    """

    def __init__(self, output, remaining_patience, **kwargs):
        """Main constructor - initializes the parent.

        output (str): path to folder where to store the models.
        remaining_patience (int): patience left when starting early stopping.
            (in general it's equal to patience - but it may be less if train is resumed)
        """
        super(EarlyStoppingWithModelSave, self).__init__(**kwargs)
        self.output = output
        self.wait = self.patience - remaining_patience

    def on_train_begin(self, logs=None):
        """See parent class doc."""
        super(EarlyStoppingWithModelSave, self).on_train_begin(logs)
        Path(self.output).mkdir(parents=True, exist_ok=True)

    # copy-pasted in order to modify what happens when we improve (see comment below)
    def on_epoch_end(self, epoch, logs=None):
        """See parent class doc."""
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

            self.model.save(os.path.join(self.output, BEST_MODEL_NAME))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

        self.model.save(os.path.join(self.output, LAST_MODEL_NAME))
        write_stats(self.output, self.best, epoch, self.patience - self.wait)
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}


def load_mlflow(output):
    """Load the mlflow run id.

    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['mlflow_run_id']


def write_mlflow(output):
    """Write the mlflow info to resume the training plotting..

    Args:
        output (str): Output directory
    """
    mlflow_run = mlflow.active_run()
    mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else 'NO_MLFLOW'
    to_store = {'mlflow_run_id': mlflow_run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def train_impl(model, datamodule, output, hyper_params,
               use_progress_bar, start_from_scratch, mlf_logger, gpus):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        datamodule (obj): lightning data module that will instantiate data loaders.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
        mlf_logger (obj): MLFlow logger callback.
        gpus: number of GPUs to use.
    """
    check_and_log_hp(['max_epoch', 'patience'], hyper_params)
    write_mlflow(output)

    best_model_path = os.path.join(output, BEST_MODEL_NAME)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=best_model_path,
        filename='model',
        save_top_k=1,
        verbose=use_progress_bar,
        monitor="val_loss",
        mode="auto",
        period=1,
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=last_model_path,
        filename='model',
        verbose=use_progress_bar,
        period=1,
    )

    resume_from_checkpoint = handle_previous_models(output, last_model_path, best_model_path,
                                                    start_from_scratch)

    early_stopping = EarlyStopping("val_loss", mode="auto", patience=hyper_params['patience'],
                                   verbose=use_progress_bar)
    trainer = pl.Trainer(
        callbacks=[early_stopping, best_checkpoint_callback, last_checkpoint_callback],
        checkpoint_callback=True,
        logger=mlf_logger,
        max_epochs=hyper_params['max_epoch'],
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpus
    )

    trainer.fit(model, datamodule=datamodule)
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    return best_dev_result


def handle_previous_models(output, last_model_path, best_model_path, start_from_scratch):
    """Moves the previous models in a new timestamp folder."""
    last_models = glob.glob(last_model_path + os.sep + '*')
    best_models = glob.glob(best_model_path + os.sep + '*')

    if len(last_models + best_models) > 0:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        new_folder = output + os.path.sep + timestamp
        os.mkdir(new_folder)
        shutil.move(last_model_path, new_folder)
        shutil.move(best_model_path, new_folder)
        logger.info(f'old models found - moving them to {new_folder}')
        # need to change the last model pointer to the new location
        last_models = glob.glob(new_folder + os.path.sep + LAST_MODEL_NAME + os.sep + '*')

    if start_from_scratch:
        logger.info('will not load any pre-existent checkpoint (because of "--start-from-scratch")')
        resume_from_checkpoint = None
    elif len(last_models) >= 1:
        resume_from_checkpoint = sorted(last_models)[-1]
        logger.info(f'models found - resuming from {resume_from_checkpoint}')
    else:
        logger.info('no model found - starting training from scratch')
        resume_from_checkpoint = None
    return resume_from_checkpoint
{%- endif %}

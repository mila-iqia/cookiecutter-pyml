import logging
import os

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
import time
import torch
import tqdm
from mlflow import log_metric
{%- endif %}
from orion.client import report_results
from yaml import dump
from yaml import load

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
STAT_FILE_NAME = 'stats.yaml'


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


def load_stats(output):
    """Load the latest statistics.

    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['best_dev_metric'], stats['epoch'], stats['remaining_patience'], \
        stats['mlflow_run_id']


def train(model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
          max_epoch, use_progress_bar=True, start_from_scratch=False):  # pragma: no cover
    """Training loop wrapper. Used to catch exception (and to handle them) if Orion is being used.

    Args:
        model (obj): The neural network model object.
        optimizer (obj): Optimizer used during training.
        loss_fun (obj): Loss function that will be optimized.
        train_loader (obj): Dataloader for the training set.
        dev_loader (obj): Dataloader for the validation set.
        patience (int): max number of epochs without improving on `best_eval_score`.
            After this point, the train ends.
        output (str): Output directory.
        max_epoch (int): Max number of epochs to train for.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
    """
    try:
        best_dev_metric = train_impl(
            model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
            max_epoch, use_progress_bar, start_from_scratch)
    except RuntimeError as err:
        if orion.client.IS_ORION_ON and 'CUDA out of memory' in str(err):
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


def train_impl(model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
               max_epoch, use_progress_bar=True, start_from_scratch=False):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        optimizer (obj): Optimizer used during training.
        loss_fun (obj): Loss function that will be optimized.
        train_loader (obj): Dataloader for the training set.
        dev_loader (obj): Dataloader for the validation set.
        patience (int): max number of epochs without improving on `best_eval_score`.
            After this point, the train ends.
        output (str): Output directory.
        max_epoch (int): Max number of epochs to train for.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
    """
    restored = reload_model(output, LAST_MODEL_NAME, start_from_scratch)

    if restored is None:
        best_dev_metric = None
        remaining_patience = patience
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
        monitor='val_loss', min_delta=0, patience=patience, verbose=use_progress_bar, mode='min',
        restore_best_weights=False, baseline=best_dev_metric, output=output,
        remaining_patience=remaining_patience
    )

    # set tensorflow/keras logging
    mt.autolog(every_n_iter=1)

    history = model.fit(x=train_loader, validation_data=dev_loader,
                        callbacks=[es], epochs=max_epoch,
                        verbose=use_progress_bar, initial_epoch=start_epoch)

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


def reload_model(output, model_name, model, optimizer,
                 start_from_scratch=False):  # pragma: no cover
    """Reload a model.

    Can be useful for model checkpointing, hyper-parameter optimization, etc.

    Args:
        output (str): Output directory.
        model_name (str): Name of the saved model.
        model (obj): A model object.
        optimizer (obj): Optimizer used during training.
        start_from_scratch (bool): starts training from scratch even if a saved moel is present.
    """
    saved_model = os.path.join(output, model_name)
    if start_from_scratch and os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - but NOT loading it '
                    '(cause --start_from_scratch)'.format(output))
        return
    if os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - loading it'.format(output))

        model.load_state_dict(torch.load(saved_model))
    if os.path.exists(output):
        logger.info('saved model file not found')
        return

    logger.info('output folder not found')
    os.makedirs(output)


def train_impl(model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
               max_epoch, use_progress_bar=True, start_from_scratch=False):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        optimizer (obj): Optimizer used during training.
        loss_fun (obj): Loss function that will be optimized.
        train_loader (obj): Dataloader for the training set.
        dev_loader (obj): Dataloader for the validation set.
        patience (int): max number of epochs without improving on `best_eval_score`.
            After this point, the train ends.
        output (str): Output directory.
        max_epoch (int): Max number of epochs to train for.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
    """
    if use_progress_bar:
        pb = tqdm.tqdm
    else:
        def pb(x, total):
            return x

    stats = reload_model(output, LAST_MODEL_NAME, model, optimizer, start_from_scratch)
    if stats is None:
        best_dev_metric = None
        remaining_patience = patience
        start_epoch = 0
    else:
        best_dev_metric, start_epoch, remaining_patience, _ = stats

    if remaining_patience <= 0:
        logger.warning(
            'remaining patience is zero - not training (and returning best dev score {})'.format(
                best_dev_metric))
        return best_dev_metric
    if start_epoch >= max_epoch:
        logger.warning(
            'start epoch {} > max epoch {} - not training (and returning best dev score '
            '{})'.format(start_epoch, max_epoch, best_dev_metric))
        return best_dev_metric
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(start_epoch, max_epoch):

        start = time.time()
        # train
        train_cumulative_loss = 0.0
        examples = 0
        model.train()
        train_steps = len(train_loader)
        for i, data in pb(enumerate(train_loader, 0), total=train_steps):
            model_input, model_target = data
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(model_input.to(device))
            model_target = torch.squeeze(model_target.to(device))
            loss = loss_fun(outputs, model_target)
            loss.backward()
            optimizer.step()

            train_cumulative_loss += loss.item()
            examples += model_target.shape[0]

        train_end = time.time()
        avg_train_loss = train_cumulative_loss / examples

        # dev
        model.eval()
        dev_steps = len(dev_loader)
        dev_cumulative_loss = 0.0
        examples = 0
        for i, data in pb(enumerate(dev_loader, 0), total=dev_steps):
            model_input, model_target = data
            with torch.no_grad():
                outputs = model(model_input.to(device))
                model_target = torch.squeeze(model_target.to(device))
                loss = loss_fun(outputs, model_target)
                dev_cumulative_loss += loss.item()
            examples += model_target.shape[0]

        avg_dev_loss = dev_cumulative_loss / examples
        log_metric("dev_loss", avg_dev_loss, step=epoch)
        log_metric("train_loss", avg_train_loss, step=epoch)

        dev_end = time.time()
        torch.save(model.state_dict(), os.path.join(output, LAST_MODEL_NAME))

        if best_dev_metric is None or avg_dev_loss < best_dev_metric:
            best_dev_metric = avg_dev_loss
            remaining_patience = patience
            torch.save(model.state_dict(), os.path.join(output, BEST_MODEL_NAME))
        else:
            remaining_patience -= 1

        logger.info(
            'done #epoch {:3} => loss {:5.3f} - dev loss {:3.2f} (will try for {} more epoch) - '
            'train min. {:4.2f} / dev min. {:4.2f}'.format(
                epoch, avg_train_loss, avg_dev_loss, remaining_patience, (train_end - start) / 60,
                (dev_end - train_end) / 60))

        write_stats(output, best_dev_metric, epoch + 1, remaining_patience)
        log_metric("best_dev_metric", best_dev_metric)

        if remaining_patience <= 0:
            logger.info('done! best dev metric is {}'.format(best_dev_metric))
            break
    logger.info('training completed (epoch done {} - max epoch {})'.format(epoch + 1, max_epoch))
    logger.info('Finished Training')
    return best_dev_metric
{%- endif %}

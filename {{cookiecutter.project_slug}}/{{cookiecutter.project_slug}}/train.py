import logging
import os

import mlflow
import orion
import yaml
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import shutil
import tensorflow as tf
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import time
import torch
import tqdm
{%- endif %}
from mlflow import log_metric
from orion.client import report_results
from yaml import dump
from yaml import load

logger = logging.getLogger(__name__)

SAVED_MODEL_NAME = 'best_model.pt'
STAT_FILE_NAME = 'stats.yaml'
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
TEMP_FOLDER = 'temp'
{%- endif %}


def reload_model(output, model, start_from_scratch=False):
    saved_model = os.path.join(output, SAVED_MODEL_NAME)
    if start_from_scratch and os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - but NOT loading it '
                    '(cause --start_from_scratch)'.format(output))
        return
    if os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - loading it'.format(output))
        model.load_state_dict(torch.load(saved_model))
        stats = load_stats(output)
        logger.info('model status: {}'.format(stats))
        return stats
    if os.path.exists(output):
        logger.info('saved model file not found - but output folder exists already - keeping it')
        return

    logger.info('no saved model file found - nor output folder - creating it')
    os.makedirs(output)

    {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
    temp_models = os.path.join(output, TEMP_FOLDER)
    if os.path.exists(saved_model):
        shutil.rmtree(temp_models)
    os.makedirs(temp_models)
    {%- endif %}


def write_stats(output, best_eval_score, epoch, remaining_patience):
    to_store = {'best_dev_metric': best_eval_score, 'epoch': epoch,
                'remaining_patience': remaining_patience,
                'mlflow_run_id': mlflow.active_run().info.run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def load_stats(output):
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['best_dev_metric'], stats['epoch'], stats['remaining_patience'], \
        stats['mlflow_run_id']


def train(model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
          max_epoch, use_progress_bar=True, start_from_scratch=False):

    try:
        best_dev_metric = train_impl(
            dev_loader, loss_fun, max_epoch, model, optimizer, output,
            patience, train_loader, use_progress_bar, start_from_scratch)
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
{%- if cookiecutter.dl_framework == 'pytorch' %}


def train_impl(dev_loader, loss_fun, max_epoch, model, optimizer, output, patience,
                       train_loader, use_progress_bar, start_from_scratch=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if use_progress_bar:
        pb = tqdm.tqdm
    else:
        pb = lambda x, total: x

    stats = reload_model(output, model, start_from_scratch)
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

    model.to(device)
    for epoch in range(start_epoch, max_epoch):

        start = time.time()
        # train
        running_loss = 0.0
        examples = 0
        model.train()
        for i, data in pb(enumerate(train_loader, 0), total=len(train_loader)):
            optimizer.zero_grad()
            model_input, model_target = data
            # forward + backward + optimize
            outputs = model(model_input.to(device))
            predictions = torch.squeeze(model_target.to(device))
            loss = loss_fun(outputs, predictions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            examples += predictions.shape[0]

        train_end = time.time()
        avg_loss = running_loss / examples

        # dev
        correct = 0
        examples = 0
        model.eval()
        with torch.no_grad():
            for i, data in pb(enumerate(dev_loader, 0), total=len(dev_loader)):
                model_input, model_target = data
                outputs = model(model_input.to(device))

                acc = (outputs > 0.5) == model_target.to(device)
                correct += torch.sum(acc).item()
                examples += model_target.shape[0]

        dev_metric = correct / examples
        log_metric("dev_metric", dev_metric, step=epoch)
        log_metric("loss", avg_loss, step=epoch)

        dev_end = time.time()

        if best_dev_metric is None or dev_metric > best_dev_metric:
            best_dev_metric = dev_metric
            remaining_patience = patience
            torch.save(model.state_dict(),
                       os.path.join(output, SAVED_MODEL_NAME))
        else:
            remaining_patience -= 1

        logger.info('done #epoch {:3} => loss {:5.3f} - dev metric {:3.2f} ('
                    'will try for {} more epoch) - train min. {:4.2f} / dev min. {:4.2f}'.format(
            epoch, running_loss / len(train_loader), dev_metric,
                   remaining_patience, (train_end - start) / 60,
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
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}


def train_impl(dev_loader, loss_fun, max_epoch, model, optimizer, output, patience, train_loader,
               use_progress_bar, start_from_scratch=False):

    stats = reload_model(output, model, start_from_scratch)
    if stats is not None:
        start_epoch = stats['epoch']
        best_dev_metric = stats['best_dev_metric']
    else:
        start_epoch = 0
        best_dev_metric = None

    metric = 'accuracy'
    metric_name = 'val_{}'.format(metric)
    es = tf.keras.callbacks.EarlyStopping(
        monitor=metric_name, min_delta=0, patience=patience, verbose=0, baseline=best_dev_metric)
    save_path = os.path.join(output, 'weights.{epoch:02d}.hdf5')
    saver = tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor=metric_name, verbose=int(use_progress_bar), save_best_only=True)

    model.compile(optimizer=optimizer, loss=loss_fun, metrics=[metric])
    history = model.fit(
        train_loader, validation_data=dev_loader, initial_epoch=start_epoch, epochs=max_epoch,
        callbacks=[es, saver], verbose=int(use_progress_bar))

    best_dev_metric = max(history.history[metric_name])
    epoch = history.epoch[-1]
    logger.info('training completed (epoch done {} - max epoch {} - best dev metric {})'.format(
        epoch + 1, max_epoch, best_dev_metric))
    log_metric("best_dev_metric", best_dev_metric)
    logger.info('Finished Training')
    write_stats(output, float(best_dev_metric), epoch + 1)

    return best_dev_metric
{%- endif %}

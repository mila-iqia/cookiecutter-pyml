import logging

{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch
from torch import optim
{%- endif %}


logger = logging.getLogger(__name__)


def load_optimizer(hyper_params, model):  # pragma: no cover
    """Instantiate the optimizer.

    Args:
        hyper_params (dict): hyper parameters from the config file
        model (obj): A neural network model object.

    Returns:
        optimizer (obj): The optimizer for the given model
    """
    optimizer_name = hyper_params['optimizer']
    # __TODO__ fix optimizer list
    if optimizer_name == 'adam':
        optimizer = {%- if cookiecutter.dl_framework == 'pytorch' %} optim.Adam(model.parameters())
    {%- else %} tf.keras.optimizers.Adam()
    {%- endif %}
    elif optimizer_name == 'sgd':
        optimizer = {%- if cookiecutter.dl_framework == 'pytorch' %} optim.SGD(model.parameters())
    {%- else %} tf.keras.optimizers.SGD()
    {%- endif %}
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_name))
    return optimizer


def load_loss(hyper_params):  # pragma: no cover
    r"""Instantiate the loss.

    You can add some math directly in your docstrings, however don't forget the `r`
    to indicate it is being treated as restructured text. For example, an L1-loss can be
    defined as:

    .. math::
        \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        loss (obj): The loss for the given model
    """
    loss_name = hyper_params['loss']
    if loss_name == 'L1':
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        loss = torch.nn.L1Loss(reduction='sum')
    {%- endif %}
    {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
        loss = tf.keras.losses.MeanAbsoluteError()
    {%- endif %}
    else:
        raise ValueError('loss {} not supported'.format(loss_name))
    return loss

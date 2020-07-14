import logging

{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch
from torch import optim
{%- endif %}

from {{cookiecutter.project_slug}}.models.my_model import MyModel

logger = logging.getLogger(__name__)


def load_model(hyper_params):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'my_model':
        model_class = MyModel
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))

    model = model_class(hyper_params)
    logger.info(model)

    {%- if cookiecutter.dl_framework == 'pytorch' %}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using device {}'.format(device))
    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_name(0))
    {%- endif %}

    return model


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
        optimizer = {%- if cookiecutter.dl_framework == 'pytorch' %} optim.Adam(model.parameters()){%- else %} tf.keras.optimizers.Adam(){%- endif %}
    elif optimizer_name == 'sgd':
        optimizer = {%- if cookiecutter.dl_framework == 'pytorch' %} optim.SGD(model.parameters()){%- else %} tf.keras.optimizers.SGD(){%- endif %}
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
    {%- if cookiecutter.dl_framework == 'pytorch' %}
    return torch.nn.L1Loss(reduction='sum')
    {%- endif %}
    {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
    return tf.keras.losses.MeanAbsoluteError()
    {%- endif %}

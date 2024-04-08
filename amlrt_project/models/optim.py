import logging

import torch

logger = logging.getLogger(__name__)


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
    loss_name = hyper_params["loss"]
    if loss_name == "cross_entropy":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("loss {} not supported".format(loss_name))
    return loss

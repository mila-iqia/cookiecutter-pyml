import random

import numpy as np
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch
{%- endif %}
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf
{%- endif %}


def set_seed(seed):  # pragma: no cover
    """Set the provided seed in python/numpy/DL framework.

    :param seed: (int) the seed
    """
    random.seed(seed)
    np.random.seed(seed)
{%- if cookiecutter.dl_framework == 'pytorch' %}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
{%- endif %}
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
    tf.random.set_seed(seed)
{%- endif %}

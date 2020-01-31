import logging

{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch.nn as nn
{%- endif %}

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class MyModel({%- if cookiecutter.dl_framework == 'pytorch' %}nn.Module{%- else %}tf.keras.Model{%- endif %}):

    def __init__(self, hyper_params):
        super(MyModel, self).__init__()

        check_and_log_hp(['size'], hyper_params)
        self.hyper_params = hyper_params

        {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
        self.hyper_params = hyper_params
        self.dense1 = tf.keras.layers.Dense(hyper_params['size'], activation=None)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        hidden1 = self.dense1(inputs)
        hidden2 = self.dense2(hidden1)
        return hidden2
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        self.linear1 = nn.Linear(5, hyper_params['size'])
        self.linear2 = nn.Linear(hyper_params['size'], 1)

    def forward(self, data):
        hidden = self.linear1(data)
        result = self.linear2(hidden)
        return result.squeeze()
        {%- endif %}

import logging

{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch.nn as nn
{%- endif %}

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class FakeModel({%- if cookiecutter.dl_framework == 'pytorch' %}nn.Module{%- else %}tf.keras.Model{%- endif %}):

    def __init__(self, hyper_params):
        super(FakeModel, self).__init__()

        check_and_log_hp(['size', 'dropout'], hyper_params)
        self.hyper_params = hyper_params

        {%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
        self.flat = tf.keras.layers.Flatten(input_shape=(5,))
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        flatten = self.flat(inputs)
        hidden1 = self.dense1(flatten)
        hidden2 = self.dense2(hidden1)
        return hidden2
        {%- endif %}
        {%- if cookiecutter.dl_framework == 'pytorch' %}
        self.linear = nn.Linear(5, 1)

    def forward(self, data):
        result = self.linear(data)
        return result.squeeze()
        {%- endif %}


    def get_hyper_params(self):
        return self.hyper_params

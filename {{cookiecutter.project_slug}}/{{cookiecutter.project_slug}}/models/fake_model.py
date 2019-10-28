import logging

import torch.nn as nn

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class FakeModel(nn.Module):

    model_name = 'fake_model'

    def __init__(self, hyper_params):
        super(FakeModel, self).__init__()

        check_and_log_hp(['size', 'dropout'], hyper_params)
        self.hyper_params = hyper_params

        self.linear = nn.Linear(5, 1)

    def forward(self, data):
        result = self.linear(data)
        return result.squeeze()

    def get_hyper_params(self):
        return self.hyper_params

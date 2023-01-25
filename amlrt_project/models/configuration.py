r"""Parse the configuration into the 4 factories.

This file is used to import the possible factories, and "register" them.
Then, the configuration dictionary is used to create instances of the correct factories.

+ Import any new models to be able to use them.
+ Add the models to the dictionary, with the name(s) you want to use in configuration.
+ The same applies for losses.
"""


import logging
from typing import Any, Dict, Optional, Type, Union

import pytorch_lightning as pl

from amlrt_project.models.losses import CrossEntropyFactory
from amlrt_project.models.optimization import (AdamFactory, OptimFactory,
                                               PlateauFactory, SchedulerFactory,
                                               SGDFactory, WarmupDecayFactory)
from amlrt_project.models.mlp import SimpleMLPFactory
from amlrt_project.models.task import ImageClassification, LossFactory, ModelFactory

logger = logging.getLogger(__name__)


OPTS = {
    'SGD': SGDFactory,
    'sgd': SGDFactory,
    'Adam': AdamFactory,
    'adam': AdamFactory
}

SCHEDS = {
    'Plateau': PlateauFactory,
    'plateau': PlateauFactory,
    'WarmupDecay': WarmupDecayFactory,
    'warmupdecay': WarmupDecayFactory
}

# TODO: Add models here
MODELS = {
    'simple_mlp': SimpleMLPFactory
}

# TODO: Add losses here, as needed.
LOSSES = {
    'cross_entropy': CrossEntropyFactory
}


def parse_opt_hp(hyper_params: Union[str, Dict[str, Any]]) -> OptimFactory:
    """Parse the optimizer part of the config."""
    if isinstance(hyper_params, str):
        algo = hyper_params
        args = {}
    elif isinstance(hyper_params, dict):
        algo = hyper_params['algo']
        args = {key: hyper_params[key] for key in hyper_params if key != 'algo'}
    else:
        raise TypeError(f"hyper_params should be a str or a dict, got {type(hyper_params)}")

    if algo not in OPTS:
        raise ValueError(f'Optimizer {algo} not supported')
    else:
        algo: Type[OptimFactory] = OPTS[algo]

    return algo(**args)


def parse_sched_hp(
    hyper_params: Optional[Union[str, Dict[str, Any]]]
) -> Optional[SchedulerFactory]:
    """Parse the scheduler part of the config."""
    if hyper_params is None:
        return None
    elif isinstance(hyper_params, str):
        algo = hyper_params
        args = {}
    elif isinstance(hyper_params, dict):
        algo = hyper_params['algo']
        args = {key: hyper_params[key] for key in hyper_params if key != 'algo'}
    else:
        raise TypeError(f"hyper_params should be a str or a dict, got {type(hyper_params)}")

    if algo not in SCHEDS:
        raise ValueError(f'Scheduler {algo} not supported')
    else:
        algo: Type[SchedulerFactory] = SCHEDS[algo]

    return algo(**args)


def parse_model_hp(hyper_params: Union[str, Dict[str, Any]]) -> ModelFactory:
    """Parse the model part of the config."""
    if isinstance(hyper_params, str):
        architecture = hyper_params
        args = {}
    elif isinstance(hyper_params, dict):
        architecture = hyper_params['architecture']
        args = {key: hyper_params[key] for key in hyper_params if key != 'architecture'}
    else:
        raise TypeError(f"hyper_params should be a str or a dict, got {type(hyper_params)}")

    if architecture not in MODELS:
        raise ValueError(f'Architecture {architecture} not supported')
    else:
        architecture: Type[ModelFactory] = MODELS[architecture]

    return architecture(**args)


def parse_loss_hp(hyper_params: Union[str, Dict[str, Any]]) -> LossFactory:
    """Parse the model part of the config."""
    if isinstance(hyper_params, str):
        loss = hyper_params
        args = {}
    elif isinstance(hyper_params, dict):
        loss = hyper_params['loss']
        args = {key: hyper_params[key] for key in hyper_params if key != 'loss'}
    else:
        raise TypeError(f"hyper_params should be a str or a dict, got {type(hyper_params)}")

    if loss not in LOSSES:
        raise ValueError(f'Loss {loss} not supported')
    else:
        loss: Type[ModelFactory] = LOSSES[loss]

    return loss(**args)


def get_model(hyper_params: Dict[str, Any]) -> pl.LightningModule:
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    # This is a patch to fit the current configuration.
    # Create a dict with the correct parameters only, and scream if they are missing.
    model_cfg = {k: hyper_params[k] for k in {'architecture', 'hidden_dim', 'num_classes'}}
    return ImageClassification(
        parse_model_hp(model_cfg),
        parse_loss_hp(hyper_params['loss']),
        parse_opt_hp(hyper_params['optimizer']),
        # Note: scheduler can be None or missing.
        parse_sched_hp(hyper_params.get('scheduler', None)))

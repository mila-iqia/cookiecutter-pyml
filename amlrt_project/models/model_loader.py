import logging
from typing import Any, Dict, Optional, Type, Union

from amlrt_project.models.factory import (AdamFactory, OptimFactory,
                                          PlateauFactory, SchedulerFactory,
                                          SGDFactory, WarmupDecayFactory)
from amlrt_project.models.my_model import SimpleMLP

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


def load_model(hyper_params: Dict[str, Any]):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'simple_mlp':
        model_class = SimpleMLP
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))

    optim_fact = parse_opt_hp(hyper_params.get('optimizer', 'SGD'))
    sched_fact = parse_sched_hp(hyper_params.get('scheduler', None))
    model = model_class(optim_fact, sched_fact, hyper_params)
    logger.info('model info:\n' + str(model) + '\n')

    return model

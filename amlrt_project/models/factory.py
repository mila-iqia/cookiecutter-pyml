"""Optimizer and learning rate scheduler factory."""


from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from torch.nn import Parameter
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


class OptimFactory(ABC):
    """Base class for optimizer factories."""

    @abstractmethod
    def __call__(self, parameters: Iterable[Parameter]) -> Optimizer:
        """Create an optimizer."""
        ...


class SchedulerFactory(ABC):
    """Base class for learning rate scheduler factories."""

    @abstractmethod
    def __call__(self, optim: Optimizer) -> Dict[str, Any]:
        """Create a scheduler."""
        ...


@dataclass
class OptimizerConfigurationFactory:
    """Combine an optimizer factory and a scheduler factory.

    Return the configuration Lightning requires.
    Only support the usual case (one optim, one scheduler.)
    """
    optim_factory: OptimFactory
    scheduler_factory: Optional[SchedulerFactory] = None

    def __call__(self, parameters: Iterable[Parameter]) -> Dict[str, Any]:
        """Create the optimizer and scheduler, for `parameters`."""
        config = {}
        optim = self.optim_factory(parameters)
        config['optimizer'] = optim
        if self.scheduler_factory is not None:
            config['lr_scheduler'] = self.scheduler_factory(optim)
        return config


@dataclass
class PlateauFactory(SchedulerFactory):
    """Reduce the learning rate when `metric` is no longer improving."""
    metric: str
    """Metric to use, must be logged with Lightning."""
    mode: str = "min"
    """Minimize or maximize."""
    factor: float = 0.1
    """Multiply the learning rate by `factor`."""
    patience: int = 10
    """Wait `patience` epoch before reducing the learning rate."""

    def __call__(self, optimizer: Optimizer) -> Dict[str, Any]:
        """Create a scheduler."""
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor, patience=self.patience)
        return dict(
            scheduler=scheduler,
            frequency=1,
            interval='epoch',
            monitor=self.metric)


@dataclass
class WarmupDecayFactory(SchedulerFactory):
    r"""Increase the learning rate linearly from zero, then decay it.

    With base learning rate $\tau$, step $s$, and `warmup` $w$, the linear warmup is:
    $$\tau \frac{s}{w}.$$
    The decay, following the warmup, is
    $$\tau \gamma^{s-w},$$ where $\gamma$ is the hold rate.
    """
    gamma: float
    r"""Hold rate; higher value decay more slowly. Limited to $\eps \le \gamma \le 1.$"""
    warmup: int
    r"""Length of the linear warmup."""
    eps: float = field(init=False, default=1e-16)
    r"""Safety value: `gamma` must be larger than this."""

    def __post_init__(self):
        """Finish initialization."""
        # Clip gamma to something that make sense, just in case.
        self.gamma = max(min(self.gamma, 1.0), self.eps)
        # Same for warmup.
        self.warmup = max(self.warmup, 0)

    def __call__(self, optimizer: Optimizer) -> Dict[str, Any]:
        """Create scheduler."""

        def fn(step: int) -> float:
            """Learning rate decay function."""
            if step < self.warmup:
                return step / self.warmup
            elif step > self.warmup:
                return self.gamma ** (step - self.warmup)
            return 1.0

        scheduler = LambdaLR(optimizer, fn)
        return dict(scheduler=scheduler, frequency=1, interval='step')


@dataclass
class SGDFactory(OptimFactory):
    """Factory for SGD optimizers."""
    lr: float = MISSING   # Value is required.
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False

    def __call__(self, parameters: Iterable[Parameter]) -> SGD:
        """Create and initialize a SGD optimizer."""
        return SGD(
            parameters, lr=self.lr,
            momentum=self.momentum, dampening=self.dampening,
            weight_decay=self.weight_decay, nesterov=self.nesterov)


@dataclass
class AdamFactory(OptimFactory):
    """Factory for ADAM optimizers."""
    lr: float = 1e-3  # `MISSING` if we want to require an explicit value.
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = True  # NOTE: The pytorch default is False, for backward compatibility.

    def __call__(self, parameters: Iterable[Parameter]) -> Adam:
        """Create and initialize an ADAM optimizer."""
        return Adam(
            parameters, lr=self.lr,
            betas=self.betas, eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad)

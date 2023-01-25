r"""Factories for loss functions.

Currently, only support cross entropy.

+ Add any loss needed, using the CrossEntropyFactory as a template.
+ Remove cross entropy if not relevant.
+ Update models.configuration to match the losses here.
"""

from dataclasses import dataclass
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CrossEntropyFactory:
    """Create a cross entropy loss function."""
    weight: Optional[torch.FloatTensor] = None
    """Optional weight per class. If present, must have one element per class,
    excluding `ignore_index`."""

    ignore_index: int = -100
    "Ignore example with this value as the target."

    label_smoothing: float = 0.0
    "If non-zero, target distribution is a mixture of uniform and the default spike."

    def __call__(self) -> torch.nn.CrossEntropyLoss:
        """Create the module with the correct hyper-parameters."""
        return torch.nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing)

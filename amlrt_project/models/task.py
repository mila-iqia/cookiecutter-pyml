r"""Define the task and how the model is trained.

Factories are used to handle the details.
This LightningModule only deal with the simple case of 1 optimizer, 1 model, 1 loss.
This should cover most cases, and should be useful as a template for more complex cases,
such as GANs.

Todo:
+ Rename the task, unless you are doing image classification.
+ Change the signature of the forward method, as needed, and update _generic_step.

This should be sufficient, since the task is only responsible for moving data around.
"""


import logging
from typing import Protocol, Tuple

import pytorch_lightning as pl

from torch import FloatTensor, LongTensor, nn

from amlrt_project.models.optimization import (OptimFactory,
                                               OptimizerConfigurationFactory,
                                               SchedulerFactory)

logger = logging.getLogger(__name__)


# TODO: Add Batch namedtuple or dataclass, and return that from the collate function.
#       This would document what is needed for the task, help avoid confusion.
#       It does add a bit of boilerplate.


class ModelFactory(Protocol):
    """Interface for the model factory.

    This is used to create the torch.nn.Module used for the task.
    The LightningModule itself is only used as the glue that move data
    around.
    """

    def __call__(self) -> nn.Module:
        """Create the model."""
        ...


# TODO: Might want to include metrics, etc in the loss, which is not supported by nn.Module.
class LossFactory(Protocol):
    """Interface for the loss factory.

    This is used to create the torch.nn.Module used for the loss.
    The LightningModule itself is only used as the glue that move data
    around.
    """

    def __call__(self) -> nn.Module:
        """Create the loss function."""
        ...


# TODO: Rename and modify this class to fit your task.
class ImageClassification(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""

    def __init__(
        self,
        model_fact: ModelFactory,
        loss_fact: LossFactory,
        optim_fact: OptimFactory,
        sched_fact: SchedulerFactory,
    ):
        """Initialize the LightningModule, with the actual model and loss."""
        super().__init__()
        self.model = model_fact()
        self.loss_fn = loss_fact()
        self.opt_fact = OptimizerConfigurationFactory(optim_fact, sched_fact)

    def configure_optimizers(self):
        """Returns the combination of optimizer(s) and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.

        This function will be called automatically by the pytorch lightning trainer implementation.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html for more info
        on the expected returned elements.
        """
        return self.opt_fact(self.model.parameters())

    # TODO: Modify the signature to fit your input / output.
    def forward(self, input_data: FloatTensor) -> FloatTensor:
        """Invoke the model."""
        return self.model(input_data)

    def _generic_step(
            self,
            batch: Tuple[FloatTensor, LongTensor],
            batch_idx: int,
    ) -> FloatTensor:
        """Runs the prediction + evaluation step for training/validation/testing."""
        input_data, targets = batch
        preds = self(input_data)  # calls the forward pass of the model
        loss = self.loss_fn(preds, targets)
        return loss

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return loss  # this function is required, as the loss returned here is used for backprop

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("test_loss", loss)

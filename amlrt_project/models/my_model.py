import logging
import typing

import pytorch_lightning as pl
from torch import nn

from amlrt_project.models.optim import load_loss, load_optimizer
from amlrt_project.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""

    def configure_optimizers(self):
        """Returns the combination of optimizer(s) and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.

        This function will be called automatically by the pytorch lightning trainer implementation.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html for more info
        on the expected returned elements.
        """
        # we use the generic loading function from the `model_loader` module, but it could be made
        # a direct part of the model (useful if we want layer-dynamic optimization)
        return load_optimizer(self.hparams, self)

    def _generic_step(
            self,
            batch: typing.Any,
            batch_idx: int,
    ) -> typing.Any:
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


class SimpleMLP(BaseModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """
    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """__init__.

        Args:
            hyper_params (dict): hyper parameters from the config file.
        """
        super(SimpleMLP, self).__init__()

        check_and_log_hp(['hidden_dim', 'num_classes'], hyper_params)
        self.save_hyperparameters(hyper_params)  # they will become available via model.hparams
        num_classes = hyper_params['num_classes']
        hidden_dim = hyper_params['hidden_dim']
        self.loss_fn = load_loss(hyper_params)  # 'load_loss' could be part of the model itself...

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential(
            nn.Linear(
                784, hidden_dim,
            ),  # The input size for the linear layer is determined by the previous operations
            nn.ReLU(),
            nn.Linear(
                hidden_dim, num_classes
            ),  # Here we get exactly num_classes logits at the output
        )

    def forward(self, x):
        """Model forward."""
        x = self.flatten(x)  # Flatten is necessary to pass from CNNs to MLP
        x = self.mlp_layers(x)
        return x

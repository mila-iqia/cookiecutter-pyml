import logging
import typing

from torch import nn
import torchmetrics
import pytorch_lightning as pl

from {{cookiecutter.project_slug}}.models.optim import load_loss, load_optimizer

from {{cookiecutter.project_slug}}.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""

    def __init__(self, hparams):
        """init."""
        super(BaseModel, self).__init__()
        self.save_hyperparameters(
            hparams, logger=True
        )  # they will become available via model.hparams and in the logger tool
        self.init_metrics()

    def init_metrics(self):
        """Initialize torchmetrics metrics.

        Note that we assume we collect the exact same metrics accross splits.
        See https://torchmetrics.readthedocs.io/en/stable/pages/overview.html for more info.
        """
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.classification.MulticlassF1Score(
                    num_classes=10, average="macro"
                ),
                torchmetrics.classification.MulticlassPrecision(
                    num_classes=10, average="macro"
                ),
                torchmetrics.classification.MulticlassRecall(
                    num_classes=10, average="macro"
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

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
        inputs, targets = batch
        logits = self(inputs)  # calls the forward pass of the model
        loss = self.loss_fn(logits, targets)
        return loss, logits, targets

    def on_train_start(self):
        """Reset train metrics."""
        self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss, logits, targets = self._generic_step(batch, batch_idx)

        metrics = self.train_metrics(logits, targets)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        self.log_dict(metrics)
        self.log("train_loss", loss)

        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return loss  # this function is required, as the loss returned here is used for backprop

    def on_validation_start(self):
        """Reset validation metrics."""
        self.val_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss, logits, targets = self._generic_step(batch, batch_idx)
        self.val_metrics.update(logits, targets)
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        """Collects the metrics at the end of validation."""
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, etc.
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss, preds, targets = self._generic_step(batch, batch_idx)


class SimpleMLP(BaseModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """
    def __init__(self, hparams: typing.Dict[typing.AnyStr, typing.Any]):
        """__init__.

        Args:
            hparams (dict): hyper parameters from the config file.
        """
        super(SimpleMLP, self).__init__(hparams)

        check_and_log_hp(["hidden_dim", "num_classes"], hparams)
        num_classes = hparams["num_classes"]
        hidden_dim = hparams["hidden_dim"]
        self.loss_fn = load_loss(
            hparams
        )  # 'load_loss' could be part of the model itself...

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential(
            nn.Linear(
                784,
                hidden_dim,
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


class SimpleCNN(BaseModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple CNN model.
    """

    def __init__(self, hparams: typing.Dict[typing.AnyStr, typing.Any]):
        """__init__.

        Args:
            hparams (dict): hyper parameters from the config file.
        """
        super(SimpleCNN, self).__init__(hparams)

        check_and_log_hp(["hidden_dim", "num_classes"], hparams)
        num_classes = hparams["hidden_dim"]
        hidden_dim = hparams["num_classes"]
        self.loss_fn = load_loss(
            hparams
        )  # 'load_loss' could be part of the model itself...

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential(
            nn.Linear(
                1568,
                hidden_dim,
            ),  # The input size for the linear layer is determined by the previous operations
            nn.ReLU(),
            nn.Linear(
                hidden_dim, num_classes
            ),  # Here we get exactly num_classes logits at the output
        )

    def forward(self, x):
        """Model forward."""
        x = self.conv_layers(x)
        x = self.flatten(x)  # Flatten is necessary to pass from CNNs to MLP
        x = self.mlp_layers(x)
        return x

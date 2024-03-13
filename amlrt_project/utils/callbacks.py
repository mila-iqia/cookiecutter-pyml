import logging

import torch
import torchmetrics
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class ComputeMetrics(Callback):
    """Keeps track of the precision, recal, and f1-score and accuracy metrics."""

    def __init__(self, num_classes: int):
        """Init."""
        self.num_classes = num_classes
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def setup(self, trainer, pl_module, stage):
        """Initial setup of the metrics."""
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task='multiclass',
                    num_classes=self.num_classes
                ),
                torchmetrics.classification.MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                torchmetrics.classification.MulticlassPrecision(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                torchmetrics.classification.MulticlassRecall(
                    num_classes=self.num_classes,
                    average="macro",
                ),
            ]
        )
        if self.train_metrics is None:
            self.train_metrics = metrics.clone(prefix="train_")

        if self.val_metrics is None:
            self.val_metrics = metrics.clone(prefix="val_")

        if self.test_metrics is None:
            self.test_metrics = metrics.clone(prefix="test_")

    def on_train_start(self, trainer, pl_module):
        """Reset train metrics."""
        self.train_metrics.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collects all samples from the batch and computes the metrics."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        probs = torch.nn.functional.softmax(logits)
        metrics = self.train_metrics(probs, targets)
        self.log_dict(metrics)

    def on_validation_start(self, trainer, pl_module):
        """Reset validation metrics."""
        self.val_metrics.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Collects all samples from the batch and computes the metrics."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        probs = torch.nn.functional.softmax(logits)
        self.val_metrics.update(probs, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Collects the metrics at the end of validation."""
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, etc.
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)


class LogConfusionMatrix(Callback):
    """Log the confusion matrix for each data split."""

    def __init__(self, num_classes):
        """init."""
        self.num_classes = num_classes
        self.train_cm = None
        self.val_cm = None
        self.test_cm = None

    def setup(self, trainer, pl_module, stage):
        """Initial setup of the confusion matrix."""
        conf_mat = torchmetrics.MetricCollection(
            torchmetrics.ConfusionMatrix(
                task='multiclass',
                num_classes=self.num_classes
            ),
        )

        if self.train_cm is None:
            self.train_cm = conf_mat.clone(prefix="train_")
        if self.val_cm is None:
            self.val_cm = conf_mat.clone(prefix="val_")
        if self.test_cm is None:
            self.test_cm = conf_mat.clone(prefix="test_")

    def on_train_start(self, trainer, pl_module):
        """Reset train metrics."""
        self.train_cm.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collects all samples from the batch and computes the metrics."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        probs = torch.nn.functional.softmax(logits)
        self.train_cm.update(probs, targets)

    def on_train_epoch_end(self, trainer, pl_module):
        """Log confusion matrix at end of train epoch."""
        conf_mats = self.train_cm.compute()
        logger.info(f"\n***Train Confusion Matrix***\n {conf_mats}")

    def on_validation_start(self, trainer, pl_module):
        """Reset validation metrics."""
        self.val_cm.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Collects all samples from the batch and computes the metrics."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        probs = torch.nn.functional.softmax(logits)
        self.val_cm.update(probs, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log confusion matrix at end of train epoch."""
        conf_mats = self.val_cm.compute()
        logger.info(f"\n***Validation Confusion Matrix***\n {conf_mats}")

    def on_test_start(self, trainer, pl_module):
        """Reset validation metrics."""
        self.test_cm.reset()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Collects all samples from the batch and computes the metrics."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        probs = torch.nn.functional.softmax(logits)
        self.test_cm.update(probs, targets)

    def on_test_epoch_end(self, trainer, pl_module):
        """Log confusion matrix at end of train epoch."""
        conf_mats = self.test_cm.compute()
        logger.info(f"\n***Test Confusion Matrix***\n {conf_mats}")

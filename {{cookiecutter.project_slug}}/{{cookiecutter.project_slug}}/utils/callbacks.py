import logging

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
import torch
import torchmetrics

logger = logging.getLogger(__name__)


def matplotlib_imshow(img, ax, single_channel=False):
    """Plot a tensor to a given matplotlib ax.

    Args:
        img (torch.tensor): a tensor which has already been normalized
        ax (matplotlib.ax): a matplotlib axis
        single_channel (bool, optional): Image is single channel or not. Defaults to False.
    """
    ax.set_axis_off()
    if single_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if single_channel:
        ax.imshow(npimg, cmap="Greys")
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_images_preds(model, images, targets, classes):
    """Plot the images and predictions to a matplotlib figure.

    Args:
        model (torch.nn.Module or pl.LightningModule): the model to do inference with
        images (torch.tensor): batch of images [N, C, H, W]
        targets (torch.tensor): batch of targets [N,]
        classes (List[Str]): the plaintext names of the classes

    Returns:
        fig (matplotlib.figure): Matplotlib figure to plot
    """
    # plot the images in the batch, along with predicted and true labels
    logits = model(images)
    probs = torch.nn.functional.softmax(logits)
    preds = torch.argmax(probs, dim=1)  # get the class prediction

    num_cols = 4
    num_rows = len(images) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for idx, image in enumerate(images):
        pred = preds[idx]
        target = targets[idx]
        pred_str = classes[pred]
        target_str = classes[target]
        prob = probs[idx][pred]

        ax = axs.ravel()[idx]
        matplotlib_imshow(image, ax, single_channel=True)

        title = f"{pred_str}, {prob*100:2.2f}%,\n (label: {target_str})"
        ax.set_title(
            title,
            color=("green" if pred == target else "red"),
        )
    return fig


class PlotValidationResults(Callback):
    """Callback to plot validation images with predictions and targets in the logger."""
    def __init__(self):
        """Define which logger we are using, eventually support many."""
        self.logger_name = "tensorboard"
        assert (
            self.logger_name == "tensorboard"
        ), "Only tensorboard is supported at this moment."

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        """Plot results of the first batch from validation in the logger."""
        classes = trainer.val_dataloaders[0].dataset.classes
        if batch_idx == 0:
            fig = plot_images_preds(
                pl_module, images=batch[0], targets=batch[1], classes=classes
            )
            if self.logger_name == "tensorboard":
                pl_module.logger.experiment.add_figure(
                    "val_pred", fig, global_step=pl_module.global_step
                )


class ComputeMetrics(Callback):
    """Keeps track of the precision, recal, and f1-score and accuracy metrics."""

    def __init__(self):
        """Init."""
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def setup(self, trainer, pl_module, stage):
        """Initial setup of the metrics."""
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
    """Keeps track of the precision, recal, and f1-score and accuracy metrics."""

    def __init__(self):
        """init."""
        self.train_cm = None
        self.val_cm = None
        self.test_cm = None

    def setup(self, trainer, pl_module, stage):
        """Initial setup of the confusion matrix."""
        conf_mat = torchmetrics.MetricCollection(
            torchmetrics.ConfusionMatrix(num_classes=10),
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

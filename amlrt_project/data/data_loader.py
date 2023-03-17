import logging
import typing
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from amlrt_project.data.data_preprocess import FashionMnistParser
from amlrt_project.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)
# __TODO__ change the dataloader to suit your needs...


class FashionMnistDS(Dataset):  # pragma: no cover
    """Dataset class for iterating over the data."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        """Initialize Dataset.

        Args:
            images (np.array): Image data [batch, height, width].
            labels (np.array): Target data [batch,].
            transform (Callable[[torch.tensor], torch.tensor], optional): Valid tensor transformations.  # noqa
            Defaults to None.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the number of data items in Dataset."""
        return len(self.images)

    def __getitem__(
        self,
        idx: int,
    ):
        """__getitem__.

        Args:
            idx (int): Get index item from the dataset.
        """
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class FashionMnistDM(pl.LightningDataModule):  # pragma: no cover
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        check_and_log_hp(["batch_size", "num_workers"], hyper_params)
        self.data_dir = data_dir
        self.batch_size = hyper_params["batch_size"]
        self.num_workers = hyper_params["num_workers"]

    def setup(self, stage=None):
        """Parses and splits all samples across the train/valid/test parsers."""
        # here, we will actually assign train/val datasets for use in dataloaders
        raw_data = FashionMnistParser(data_dir=self.data_dir)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = FashionMnistDS(
                raw_data.train_images, raw_data.train_labels, transform
            )
            self.val_dataset = FashionMnistDS(
                raw_data.val_images, raw_data.val_labels, transform
            )
        if stage == "test" or stage is None:
            self.test_dataset = FashionMnistDS(
                raw_data.test_images, raw_data.test_labels, transform
            )

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Creates the validation dataloader using the validation data parser."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

import gzip
import logging
import os
import typing
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)


BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


class FashionMnistParser:
    """Parser for the Fashion MNIST dataset.

    This parser will download the original data and process it to numpy arrays.
    It will also generate train/val/test splits.
    """
    def __init__(self, data_dir):
        """Processing occurs on init."""
        self.data_dir = data_dir
        self.prepare_dataset()

    @staticmethod
    def download_dataset(data_dir: typing.AnyStr):
        """Download and extract the fashion mnist dataset to data_dir."""
        files = [
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
        ]
        logger.info("Fetching fashion mnist...")
        # Create dataset dir if it doesn't already exist
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        for fname in files:
            url = BASE_URL + fname
            output_fname = os.path.join(data_dir, fname)
            if os.path.isfile(output_fname):
                logger.info(f"{fname} already downloaded. Skipping {data_dir}")
                continue
            logger.info(f"downloading {fname} to {data_dir}")
            urllib.request.urlretrieve(url, output_fname)

    @staticmethod
    def extract_images(fname: typing.AnyStr):
        """Extract raw bytes to numpy arrays.

        See: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python # noqa
        """
        with gzip.open(fname, "r") as f:
            # skip first 4 bytes
            _ = int.from_bytes(f.read(4), "big")
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), "big")
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), "big")
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), "big")
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(
                (image_count, row_count, column_count)
            )
            # images = np.expand_dims(images, axis=-1) # add greyscale color channel
            return images

    @staticmethod
    def extract_labels(fname: typing.AnyStr):
        """Extract the labels [0-9] for the dataset."""
        with gzip.open(fname, "r") as f:
            # skip first 4 bytes
            _ = int.from_bytes(f.read(4), "big")
            # second 4 bytes is the number of labels
            _ = int.from_bytes(f.read(4), "big")
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            return labels

    @staticmethod
    def val_from_train(images: np.ndarray, labels: np.ndarray, val_pct: float):
        """Fashion mnist doesn't have a validation set, we create one here."""
        assert 0 < val_pct < 1
        num_samples = len(images)
        train_pct = 1 - val_pct
        train_idx = int(num_samples * train_pct)

        train_images = images[0:train_idx]
        train_labels = labels[0:train_idx]

        val_images = images[train_idx:]
        val_labels = labels[train_idx:]

        return train_images, train_labels, val_images, val_labels

    @staticmethod
    def subsample_dataset(images: np.ndarray, labels: np.ndarray, num_samples: int):
        """Extract a subset of the dataset to speed up training."""
        return images[:num_samples], labels[:num_samples]

    def prepare_dataset(self):
        """Download, processes and splits the data."""
        data_dir = self.data_dir
        self.download_dataset(data_dir=data_dir)

        # load the train set, split to train and val
        images = self.extract_images(
            os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        )
        labels = self.extract_labels(
            os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        )

        images, labels = self.subsample_dataset(images, labels, num_samples=20000)
        train_images, train_labels, val_images, val_labels = self.val_from_train(
            images, labels, val_pct=0.2
        )

        test_images = self.extract_images(
            os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
        )
        test_labels = self.extract_labels(
            os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
        )

        self.train_images = train_images
        self.train_labels = train_labels

        self.val_images = val_images
        self.val_labels = val_labels

        self.test_images = test_images
        self.test_labels = test_labels

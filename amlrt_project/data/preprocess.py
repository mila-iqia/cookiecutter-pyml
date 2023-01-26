"""Process the data on the disk and load it in memory."""

import gzip
import logging
import os
import typing
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)


BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
DATA = {
    'train': 'train-{type:s}-idx3-ubyte.gz',
    'test': 't10k-{type:s}-idx3-ubyte.gz'
}
TRAIN_VAL_SAMPLES = 20000
VAL_RATIO = 0.2


class Data(typing.NamedTuple):
    """Images with labels."""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, item: typing.Union[int, slice]) -> "Data":
        """Get slice or item."""
        if isinstance(item, int):
            item = slice(item, item + 1)
        elif not isinstance(item, slice):
            raise TypeError()

        return Data(self.images[item], self.labels[item])

    @classmethod
    def load(cls, ftemplate: str) -> "Data":
        """Load images from a templated file names pair."""
        images = extract_images(ftemplate.format(type='images'))
        labels = extract_labels(ftemplate.format(type='labels'))
        return Data(images, labels)


def download_dataset(data_dir: str):
    """Download and extract the fashion mnist dataset to data_dir."""
    files = (
        [fname.format(type='images') for fname in DATA.values()]
        + [fname.format(type='labels') for fname in DATA.values()])
    logger.info("Fetching fashion mnist...")
    logger.debug(f"Fetching fashion mnist from {BASE_URL}")

    # Create dataset dir if it doesn't already exist
    os.makedirs(data_dir)

    for fname in files:
        url = BASE_URL + fname
        output_fname = os.path.join(data_dir, fname)
        if os.path.isfile(output_fname):
            logger.info(f"{fname} already downloaded.")
            continue
        logger.info(f"downloading {fname} to {data_dir}")
        urllib.request.urlretrieve(url, output_fname)


def load_train_val(
    data_dir: str,
    sample=TRAIN_VAL_SAMPLES,
    ratio=VAL_RATIO
) -> typing.Tuple[Data, Data]:
    """Load the training data, and create the train-val split."""
    data = Data.load(os.path.join(data_dir, DATA['train']))
    data = data[:sample]

    return val_from_train(data, ratio)


def load_test(data_dir: str) -> Data:
    """Load the test data."""
    return Data.load(os.path.join(data_dir, DATA['test']))


def extract_images(fname: str) -> np.ndarray:
    """Extract raw bytes to numpy arrays of images.

    Args:
        fname: source file name, assumed to be gzip.

    Returns:
        images: array of `count` images, with shape `[count, height, width]`.

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


def extract_labels(fname: str) -> np.ndarray:
    """Extract the labels [0-9] for the dataset.

    Args:
        fname: source file name, assumed to be gzip.

    Returns:
        images: array of `count` labels, with shape `[count]`.
    """
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


def val_from_train(
    data: Data,
    val_pct: float
) -> typing.Tuple[Data, Data]:
    """Fashion mnist doesn't have a validation set, we create one here.

    Args:
        data: Data to split.
        val_pct: Validation ratio.

    Returns:
        train: Training split
        val: Validation split
    """
    assert 0 < val_pct < 1
    num_samples = len(data)
    train_pct = 1 - val_pct
    train_idx = int(num_samples * train_pct)

    train = data[:train_idx]
    val = data[train_idx:]

    return train, val

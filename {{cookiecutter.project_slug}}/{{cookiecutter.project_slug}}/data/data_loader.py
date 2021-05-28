import os
import typing

import numpy as np
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf

BUFFER_SIZE = 100
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
{%- endif %}
# __TODO__ change the dataloader to suit your needs...


def get_data(
    data_folder: typing.AnyStr,
    prefix: typing.AnyStr
) -> typing.Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Function to load data into memory.

    Args:
        data_folder (str): Path of the folder where the data lives.
        prefix (str): The data split to target, i.e. "train" or "dev.

    Returns:
        in_data (np.array): Input data.
        tar_data (np.array): Target data.
    """
    inputs = []
    with open(os.path.join(data_folder, '{}.input'.format(prefix))) as in_stream:
        for line in in_stream:
            inputs.append([float(x) for x in line.split()])
    in_data = np.array(inputs, dtype=np.float32)
    targets = []
    with open(os.path.join(data_folder, '{}.target'.format(prefix))) as in_stream:
        for line in in_stream:
            targets.append(float(line))
    tar_data = np.array(targets, dtype=np.float32)
    return in_data, tar_data
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}


def load_data(data_dir, hyper_params):  # pragma: no cover
    """Prepare the data into datasets.

    Args:
        data_dir (str): path to the folder containing the data
        hyper_params (dict): hyper parameters from the config file

    Retruns:
        train_dataset (obj): iterable training dataset object
        dev_dataset (obj): iterable validation dataset object


    """
    # __TODO__ load the data
    train_examples, train_labels = get_data(data_dir, 'train')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    dev_examples, dev_labels = get_data(data_dir, 'dev')
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_examples, dev_labels))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(batch_size=hyper_params['batch_size'])
    dev_dataset = dev_dataset.batch(batch_size=hyper_params['batch_size'])
    return train_dataset, dev_dataset
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}


class MyDataset(Dataset):  # pragma: no cover
    """Dataset class for iterating over the data."""

    def __init__(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
    ):
        """Initialize MyDataset.

        Args:
            input_data (np.array): Input data.
            target_data (np.array): Target data.
        """
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        """Return the number of data items in MyDataset."""
        return len(self.input_data)

    def __getitem__(
        self,
        index: int,
    ):
        """__getitem__.

        Args:
            index (int): Get index item from the dataset.
        """
        target_example = self.target_data[index]
        input_example = self.input_data[index]
        return input_example, target_example


class MyDataModule(pl.LightningDataModule):  # pragma: no cover
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = hyper_params['batch_size']
        self.train_data_parser, self.dev_data_parser = None, None

    def prepare_data(self):
        """Downloads/extracts/unpacks the data if needed (we don't)."""
        pass

    def setup(self, stage=None):
        """Parses and splits all samples across the train/valid/test parsers."""
        # here, we will actually assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_input, train_target = get_data(self.data_dir, 'train')
            self.train_data_parser = MyDataset(train_input, train_target)
            dev_input, dev_target = get_data(self.data_dir, 'dev')
            self.dev_data_parser = MyDataset(dev_input, dev_target)
        if stage == 'test' or stage is None:
            raise NotImplementedError  # __TODO__: add code to instantiate the test data parser here

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser."""
        return DataLoader(self.train_data_parser, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Creates the validation dataloader using the validation data parser."""
        return DataLoader(self.dev_data_parser, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        raise NotImplementedError  # __TODO__: add code to instantiate the test data loader here


def load_data(data_dir, hyper_params):  # pragma: no cover
    """Prepare the data into datasets.

    Args:
        data_dir (str): path to the folder containing the data
        hyper_params (dict): hyper parameters from the config file

    Returns:
        datamodule (obj): the data module used to prepare/instantiate data loaders.
    """
    # __TODO__ if you have different data modules, add whatever code is needed to select them here
    return MyDataModule(data_dir, hyper_params)
{%- endif %}

import os

import numpy
{%- if cookiecutter.dl_framework in ['tensorflow_cpu', 'tensorflow_gpu'] %}
import tensorflow as tf

BUFFER_SIZE = 100
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
from torch.utils.data import Dataset, DataLoader
{%- endif %}
# __TODO__ change the dataloader to suit your needs...


def get_data(data_folder, prefix):  # pragma: no cover
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
    in_data = numpy.array(inputs, dtype=numpy.float32)
    targets = []
    with open(os.path.join(data_folder, '{}.target'.format(prefix))) as in_stream:
        for line in in_stream:
            targets.append(float(line))
    tar_data = numpy.array(targets, dtype=numpy.float32)
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

    def __init__(self, input_data, target_data):
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

    def __getitem__(self, index):
        """__getitem__.

        Args:
            index (int): Get index item from the dataset.
        """
        target_example = self.target_data[index]
        input_example = self.input_data[index]
        return input_example, target_example


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
    train_input, train_target = get_data(data_dir, 'train')
    train_data = MyDataset(train_input, train_target)
    dev_input, dev_target = get_data(data_dir, 'dev')
    dev_data = MyDataset(dev_input, dev_target)
    train_loader = DataLoader(train_data, batch_size=hyper_params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=hyper_params['batch_size'], shuffle=False)
    return train_loader, dev_loader
{%- endif %}

import numpy
{%- if cookiecutter.dl_framework == 'tensorflow' %}
import tensorflow as tf


def get_data():
    amount = 100
    in_data = numpy.random.rand(amount, 5)
    target = numpy.random.choice([True, False], amount, replace=True, p=[0.5, 0.5])
    return in_data, target


def load_data(args, hyper_params):
    # __TODO__ load the data
    train_examples, train_labels = get_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    dev_examples, dev_labels = get_data()
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_examples, dev_labels))
    train_dataset = train_dataset.shuffle(True).batch(batch_size=hyper_params['batch_size'])
    dev_dataset = dev_dataset.batch(batch_size=hyper_params['batch_size'])
    return train_dataset, dev_dataset
{%- endif %}
{%- if cookiecutter.dl_framework == 'pytorch' %}
import torch
from torch.utils.data import Dataset, DataLoader


# __TODO__ add a data loader that makes sense.
class FakeDataset(Dataset):

    def __init__(self):
        self.amount = 100
        in_data = numpy.random.rand(self.amount, 5)
        self.in_data = torch.FloatTensor(in_data.astype('float'))
        target = numpy.random.choice([True, False], self.amount,
                                     replace=True, p=[0.5, 0.5])
        self.target = torch.FloatTensor(target.astype('float'))

    def __len__(self):
        return self.amount

    def __getitem__(self, index):
        target = self.target[index]
        data_val = self.in_data[index]
        return data_val, target


def load_data(args, hyper_params):
    # __TODO__ load the data
    train_data = FakeDataset()
    dev_data = FakeDataset()
    train_loader = DataLoader(train_data, batch_size=hyper_params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=hyper_params['batch_size'], shuffle=False)
    return train_loader, dev_loader
{%- endif %}
import numpy
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

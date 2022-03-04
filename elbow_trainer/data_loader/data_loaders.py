from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils import data
import numpy as np
import h5py
import os
import torch


class SingleH5Dataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.h5file = h5py.File(path, 'r')
        self.data = self.h5file['images']
        self.labels = self.h5file['masks']
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        labels = self.labels[index]

        if self.img_transform is not None:
            datum = self.img_transform(datum)
        labels = torch.Tensor(np.moveaxis(labels, -1, 0))
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return datum, labels

    def __len__(self):
        return self.labels.shape[0]

    def close(self):
        self.h5file.close()


class ElbowDataLoader(BaseDataLoader):
    """
    elbow data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        img_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(128)
        ])
        label_trsfm = transforms.Compose([
            transforms.CenterCrop(128)
        ])
        self.data_dir = data_dir
        list_of_datasets = []
        i = 0
        for file in os.listdir(self.data_dir):
            filename = os.fsdecode(file)
            if filename.endswith(".h5"):
                list_of_datasets.append(
                    SingleH5Dataset(self.data_dir + filename, img_transform=img_trsfm, label_transform=label_trsfm))
            if i > 3:
                break
            i = i + 1

        self.dataset = data.ConcatDataset(list_of_datasets)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

from datasets import CasiaSurfDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


def get_loader(dataset, batch_size=64, split=(4, 1)):
    val_q = len(dataset) // sum(split) * split[1]

    val_indices = np.random.choice(list(range(len(dataset))), val_q)
    train_indices = list(set(range(len(dataset))).difference(val_indices))
    sampler = SubsetRandomSampler(train_indices)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


if __name__ == '__main__':
    dataset = CasiaSurfDataset()
    data_loader = get_loader(dataset)
    for images, labels in enumerate(data_loader):
        print(images, labels)
        raise

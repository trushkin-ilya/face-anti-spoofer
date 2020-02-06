import numpy as np
from torch.utils import data


class SplittedDataLoader:
    def __init__(self, dataset: data.Dataset, train_batch_size: int, val_batch_size: int, split=(4, 1), num_workers=0):
        val_q = len(dataset) // sum(split) * split[1]

        np.random.seed(2020)
        val_indices = np.random.choice(list(range(len(dataset))), val_q)
        train_indices = list(set(range(len(dataset))).difference(val_indices))
        train_sampler = data.SubsetRandomSampler(train_indices)
        vaild_sampler = data.SubsetRandomSampler(val_indices)

        self.train = data.DataLoader(
            dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=num_workers)
        self.val = data.DataLoader(
            dataset, sampler=vaild_sampler, batch_size=val_batch_size, num_workers=num_workers)

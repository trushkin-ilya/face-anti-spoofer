from datasets import MfsdDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils import model_zoo
import numpy as np

if __name__ == '__main__':
    dataset = MfsdDataset("../data/MFSD/attack/001", "../data/MFSD/real/001")
    val_q = len(dataset) // 5
    print(val_q, range(len(dataset)))

    val_indices = np.random.choice(list(range(len(dataset))), val_q)
    train_indices = list(set(range(len(dataset))).difference(val_indices))
    train_loader = DataLoader(
        dataset, sampler=SubsetRandomSampler(train_indices), batch_size=64)
    detector = model_zoo.load_url(
        'https://docs.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1', model_dir='models')
    for i, batch in train_loader:
        print(i, batch)

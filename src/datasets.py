import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
import torch
from PIL import Image


class MfsdDataset(Dataset):
    """Mfsd dataset."""

    def __init__(self, attack_dir, real_dir):
        """
        Args:
            dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attack_dir = attack_dir
        self.real_dir = real_dir
        self.images = [{'image': os.path.join(attack_dir, img),
                        'label': 'spoof'} for img in os.listdir(attack_dir)]
        self.images = self.images + [{'image': os.path.join(real_dir, img),
                                      'label': 'live'}
                                     for img in os.listdir(real_dir)]
        self.transform = Compose([Resize((480, 640)), ToTensor()])

    def __len__(self):
        return len(os.listdir(self.attack_dir)) +\
            len(os.listdir(self.real_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]['image']
        image = Image.open(img_name)
        image = self.transform(image)

        return image, self.images[idx]['label']


class CasiaSurfDataset(Dataset):
    def __init__(self, dir: str = 'data/CASIA_SURF', train: bool = True):
        txt_metadata = 'train' if train else 'dev_res'
        self.dir = dir
        self.items = []
        for i in [1, 2, 3]:
            file_name = f'4@{i}_{txt_metadata}.txt'
            with open(os.path.join(dir, file_name), 'r') as file:
                lines = file.readlines()
                self.items += [tuple(line.split(' ')) for line in lines]
        self.transform = Compose([Resize((640, 480)), ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.items[idx][0]
        label = self.items[idx][1]
        img_path = os.path.join(self.dir, img_name)
        img = Image.open(img_path)
        img = self.transform(img)

        return img, label

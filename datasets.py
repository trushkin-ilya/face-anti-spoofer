import os
import re
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MfsdDataset(Dataset):
    """Mfsd dataset."""

    def __init__(self, attack_dir, real_dir, transform=None):
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
        self.images += [{'image': os.path.join(real_dir, img),
                         'label': 'live'}
                        for img in os.listdir(real_dir)]
        self.transform = transforms.Compose([transform, transforms.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]['image']
        image = Image.open(img_name)
        image = self.transform(image)

        return image, self.images[idx]['label']


class CasiaSurfDataset(Dataset):
    def __init__(self, protocol: int, dir: str = 'data/CASIA_SURF', mode: str = 'train', transform=None):
        self.dir = dir
        self.mode = mode
        submode = {'train': 'train', 'dev': 'dev_ref', 'test': 'test_res'}[mode]
        file_name = f'4@{protocol}_{submode}.txt'
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            if self.mode == 'train':
                self.items = [tuple(line[:-1].split(' ')) for line in lines]
            else:
                self.items = []
                for line in lines:
                    path, label = line[:-1].split(' ')
                    dir_name = os.path.join(path, 'profile')
                    for file_name in os.listdir(os.path.join(dir, dir_name)):
                        item = os.path.join(dir_name, file_name)
                        self.items.append((item, -1 if self.mode == 'test' else label))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.items[idx][0]
        label = self.items[idx][1]
        img_path = os.path.join(self.dir, img_name)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, int(label)

    def get_video_id(self, idx: int):
        img_name = self.items[idx][0]
        return re.search(rf'(?P<id>{self.mode}/\d+)', img_name).group('id')

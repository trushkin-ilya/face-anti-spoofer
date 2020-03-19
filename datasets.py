import random
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
    def __init__(self, protocol: int, dir: str = 'data/CASIA_SURF', mode: str = 'train', depth=True, ir=True,
                 transform=None):
        self.dir = dir
        self.mode = mode
        submode = {'train': 'train', 'dev': 'dev_ref', 'test': 'test_res'}[mode]
        file_name = f'4@{protocol}_{submode}.txt'
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            self.items = []
            for line in lines:
                if self.mode == 'train':
                    img_name, label = tuple(line[:-1].split(' '))
                    self.items.append((self.get_all_modalities(img_name, depth, ir), label))
                elif self.mode == 'dev':
                    folder_name, label = tuple(line[:-1].split(' '))
                    profile_dir = os.path.join(self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append((self.get_all_modalities(img_name, depth, ir), label))
                elif self.mode == 'test':
                    folder_name = line[:-1].split(' ')[0]
                    profile_dir = os.path.join(self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append((self.get_all_modalities(img_name, depth, ir), -1))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names, label = self.items[idx]
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.dir, img_name)
            img = Image.open(img_path).convert('RGB' if 'profile' in img_path else 'L')
            if self.transform:
                random.seed(idx)
                img = self.transform(img)
            images += [img]

        return images, int(label)

    def get_video_id(self, idx: int):
        img_name = self.items[idx][0]
        return re.search(rf'(?P<id>{self.mode}/\d+)', img_name).group('id')

    def get_all_modalities(self, img_path: str, depth: bool = True, ir: bool = True) -> list:
        result = [img_path]
        if depth:
            result += [re.sub('profile', 'depth', img_path)]
        if ir:
            result += [re.sub('profile', 'ir', img_path)]

        return result

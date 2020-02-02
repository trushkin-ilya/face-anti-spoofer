import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
import torch
from PIL import Image
import re


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
        self.transform = Compose([Resize((320, 240)), ToTensor()])

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
    def __init__(self, protocol: int, dir: str = '../data/CASIA_SURF', train: bool = True, ):
        txt_metadata = 'train' if train else 'dev_res'
        self.dir = dir
        file_name = f'4@{protocol}_{txt_metadata}.txt'
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            if train:
                self.items = [tuple(line[:-1].split(' ')) for line in lines]
            else:
                self.items = []
                for line in lines:
                    dir_name = os.path.join(line[:-1], 'profile')
                    for file_name in os.listdir(os.path.join(dir, dir_name)):
                        item = os.path.join(dir_name, file_name)
                        self.items.append((item, -1))

        self.transform = Compose([Resize((320, 240)), ToTensor()])

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

        return img, int(label)

    def get_video_id(self, idx: int):
        img_name = self.items[idx][0]
        return re.search(r'(?P<id>dev/\d+)', img_name).group('id')

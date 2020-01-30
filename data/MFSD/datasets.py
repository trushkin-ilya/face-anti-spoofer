import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torch
from PIL import Image


class MfsdDataset(Dataset):
    """Mfsd dataset."""

    def __init__(self, attack_dir, real_dir, transform=Resize((480, 640))):
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
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.attack_dir)) +\
            len(os.listdir(self.real_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]['image']
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.images[idx]['label']

from baseline.datasets import CasiaSurfDataset
from torchvision import transforms
from torch import optim, nn
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import models
import numpy as np
from tqdm import tqdm
from baseline.datasets import NonZeroCrop


def get_loss(lr, dataloader, args):
    model = getattr(models, args.model)(num_classes=args.num_classes)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for i, (images, labels) in enumerate(dataloader):
        if i > 4:
            return np.mean(losses)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--model', required=True, type=str)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--ir', type=bool, default=False)
    args = argparser.parse_args()

    dataset = CasiaSurfDataset(protocol=3, mode='train', depth=False, ir=False, transform=transforms.Compose(
        [NonZeroCrop(),
         transforms.Resize(256),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]))
    dataloader = data.DataLoader(
        dataset, sampler=data.sampler.RandomSampler(dataset), batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.figure(figsize=(21, 9))
    plt.xscale("log")
    plt.xlim((3e-10, 3e-1))
    lrs = np.logspace(start=1, stop=10, base=0.1, num=100)

    print(lrs, [get_loss(lr, dataloader, args)
                for lr in tqdm(lrs)])

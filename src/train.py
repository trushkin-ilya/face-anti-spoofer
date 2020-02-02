from datasets import CasiaSurfDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
from torch import optim, nn
from torchvision import models
import argparse


def get_loader(dataset, batch_size=8, split=(4, 1)):
    val_q = len(dataset) // sum(split) * split[1]

    val_indices = np.random.choice(list(range(len(dataset))), val_q)
    train_indices = list(set(range(len(dataset))).difference(val_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    vaild_sampler = SubsetRandomSampler(val_indices)
    return DataLoader(dataset, sampler=train_sampler, batch_size=batch_size),\
        DataLoader(dataset, sampler=vaild_sampler, batch_size=batch_size)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--checkpoint', type=str)
    argparser.add_argument('--batch_size', type=int, required=True)
    argparser.add_argument('--eval_every', type=int, default=1)
    args = argparser.parse_args()
    dataset = CasiaSurfDataset(args.protocol)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_loader(dataset)
    model = models.mobilenet_v2(
        num_classes=2) if not args.checkpoint else torch.load(args.checkpoint)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    print(model)
    for epoch in range(args.epochs):
        model.train()

        for i, batch in enumerate(train_loader):
            images, labels = batch
            labels = torch.LongTensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            print(
                f'Epoch: {epoch + 1}/{args.epochs}\tBatch: {i + 1}/{len(train_loader)}\tLoss: {loss.item()}')
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict,
                   f'checkpoints/mobilenet_v2_protocol{args.protocol}({epoch}).pt')

        if epoch % args.eval_every == 1:
            model.eval()
            print("Evaluating...")
            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    images, labels = batch
                    labels = torch.LongTensor(labels)
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    acc = (torch.max(outputs.data, 1) == labels).sum().item()
                    val_acc.append(acc.item())
                    val_loss.append(loss.item())
                print(
                    f"\t\t\tValidation loss: {np.mean(val_loss)}\t accuracy: {np.mean(val_acc)}")

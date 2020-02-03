from datasets import CasiaSurfDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
from torch import optim, nn
from torchvision import models, transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import os


def get_loader(dataset, batch_size=8, split=(4, 1)):
    val_q = len(dataset) // sum(split) * split[1]

    np.random.seed(2020)
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
    argparser.add_argument('--save_path', type=str, required=True)
    argparser.add_argument('--lr', type=float, default=3e-3)
    argparser.add_argument('--num_classes', type=int, default=2)
    args = argparser.parse_args()
    dataset = CasiaSurfDataset(
        args.protocol, transform=transforms.Resize((320, 240)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_loader(dataset)
    model = models.mobilenet_v2(
        num_classes=args.num_classes) if not args.checkpoint else torch.load(args.checkpoint)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()
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
        file_name = f'mobilenet_v2_protocol{args.protocol}({epoch}).pt'
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(model.state_dict, os.path.join(args.save_path, file_name))

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
            avg_loss = np.mean(val_loss)
            avg_acc = np.mean(val_acc)
            print(
                f"\t\t\tValidation loss: {avg_loss}\t accuracy: {avg_acc}")
            writer.add_scalar('Validation loss', avg_loss, epoch)
            writer.add_scalar('Validation accuracy', avg_acc, epoch)
    writer.close()

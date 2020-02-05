from datasets import CasiaSurfDataset
import numpy as np
import torch
from torch import optim, nn
from torchvision import models, transforms
import argparse
from test import evaluate
import os
import utils
    

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for i, batch in enumerate(dataloader):
        images, labels = batch
        labels = torch.LongTensor(labels)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        print(f'Epoch: {epoch + 1}/{args.epochs}\tBatch: {i + 1}/{len(train_loader)}\tLoss: {loss.item()}')
        loss.backward()
        optimizer.step()



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
    argparser.add_argument('--save_every', type=int, default=1)
    args = argparser.parse_args()

    dataset = CasiaSurfDataset(args.protocol, transform=transforms.Resize((320, 240)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = utils.SplittedDataLoader(dataset, args.batch_size)
    model = models.mobilenet_v2(num_classes=args.num_classes) if not args.checkpoint else torch.load(args.checkpoint)
    model = model.to(device)
    print(model)

    for epoch in range(args.epochs):
        train(model,
        dataloader=dataloader.train,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer = optim.Adam(model.parameters(), lr=args.lr))

        if epoch % args.save_every == 0 and epoch > 0:
            file_name = f'mobilenet_v2_protocol{args.protocol}({epoch}).pt'
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict, os.path.join(args.save_path, file_name))

        if epoch % args.eval_every == 1:
            evaluate(dataloader.val, model, loss_fn)
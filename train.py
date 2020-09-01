import os
import argparse
import torch
import numpy as np
import models, optimizers, losses
import yaml

from baseline.datasets import CasiaSurfDataset
from torch.utils import tensorboard, data
from test import evaluate
from transforms import ValidationTransform, TrainTransform


def train(model, dataloader, loss_fn, optimizer, callback):
    model.train()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images, labels = batch
        labels = torch.LongTensor(labels)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        print(
            f'Epoch: {epoch + 1}/{args.epochs}\t',
            f'Batch: {i + 1}/{len(dataloader)}\t',
            f'Loss: {loss.item()}')
        loss.backward()
        optimizer.step()


def validation_callback(model, loader, writer, epoch):
    apcer, bpcer, acer = evaluate(loader, model)
    print(f"\t\t\tAPCER: {apcer}\t BPCER: {bpcer}\t ACER: {acer}")
    writer.add_scalar('APCER', apcer, epoch)
    writer.add_scalar('BPCER', bpcer, epoch)
    writer.add_scalar('ACER', acer, epoch)
    return acer


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--config-path', type=str, required=True)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--checkpoint', type=str)
    argparser.add_argument('--train_batch_size', type=int, default=1)
    argparser.add_argument('--val_batch_size', type=int, default=1)
    argparser.add_argument('--eval_every', type=int, default=1)
    argparser.add_argument('--save_path', type=str, default='checkpoints')
    argparser.add_argument('--save_every', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--data_dir', type=str, required=True)
    args = argparser.parse_args()
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, config['model'])(num_classes=config['num_classes'])

    val_data = CasiaSurfDataset(args.protocol, dir=args.data_dir, mode='dev', depth=config['depth'], ir=config['ir'],
                                transform=ValidationTransform())

    train_data = torch.utils.data.ConcatDataset(
        [CasiaSurfDataset(protocol, dir=args.data_dir, mode='train', depth=config['depth'], ir=config['ir'],
                          transform=TrainTransform()) for protocol in [1, 2, 3]])

    train_loader = data.DataLoader(train_data, batch_size=args.train_batch_size, num_workers=args.num_workers,
                                   sampler=data.RandomSampler(train_data))
    val_loader, sub_val_loader = [data.DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        sampler=sampler) for sampler in [None, data.SubsetRandomSampler(np.random.choice(len(val_data), size=1000))]]

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    print(model)
    writer = tensorboard.SummaryWriter()
    optimizer = getattr(optimizers, config['optimizer']['class'])(model.parameters(),
                                                                  lr=float(config['optimizer']['lr']))
    scheduler = getattr(optimizers.lr_scheduler, config['lr_scheduler'])(optimizer)

    for epoch in range(args.epochs):
        train(model,
              dataloader=train_loader,
              loss_fn=getattr(losses, config['loss_fn'])(),
              optimizer=optimizer,
              callback=lambda i: validation_callback(model, sub_val_loader, writer, i))

        if epoch % args.save_every == 0:
            file_name = f'{model.__class__.__name__}_p{args.protocol}({epoch}).pt'
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                args.save_path, file_name))

        if epoch % args.eval_every == 0:
            acer = validation_callback(model, val_loader, writer, epoch)
            scheduler.step(acer)
            writer.add_scalar(
                'Learning rate', optimizer.param_groups[0]['lr'], epoch)
    writer.close()

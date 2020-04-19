import os
import argparse
import torch

from baseline.datasets import CasiaSurfDataset
from torch import optim, nn
from torchvision import models, transforms
from torch.utils import tensorboard, data
from test import evaluate


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for i, batch in enumerate(dataloader):
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--checkpoint', type=str)
    argparser.add_argument('--train_batch_size', type=int, default=1)
    argparser.add_argument('--val_batch_size', type=int, default=1)
    argparser.add_argument('--eval_every', type=int, default=1)
    argparser.add_argument('--save_path', type=str, required=True)
    argparser.add_argument('--lr', type=float, default=3e-3)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--save_every', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--data_dir', type=str, required=True)
    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    val_data = CasiaSurfDataset(args.protocol, dir=args.data_dir, mode='dev', depth=False, ir=False,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()]))

    train_data = torch.utils.data.ConcatDataset(
        [CasiaSurfDataset(protocol, dir=args.data_dir, mode='train', depth=False, ir=False,
                          transform=transforms.Compose([
                              transforms.Resize(256),
                              transforms.RandomCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()])) for protocol in [1, 2, 3]])

    train_loader = data.DataLoader(train_data, batch_size=args.train_batch_size, num_workers=args.num_workers,
                                   sampler=data.RandomSampler(train_data))
    val_loader = data.DataLoader(val_data, batch_size=args.val_batch_size, num_workers=args.num_workers)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    print(model)
    writer = tensorboard.SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(args.epochs):
        train(model,
              dataloader=train_loader,
              loss_fn=nn.CrossEntropyLoss(),
              optimizer=optimizer)

        if epoch % args.save_every == 0:
            file_name = f'MobileLiteNet54_se_p{args.protocol}({epoch}).pt'
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                args.save_path, file_name))

        if epoch % args.eval_every == 0:
            apcer, bpcer, acer = evaluate(val_loader, model)
            scheduler.step(acer)
            print(
                f"\t\t\tAPCER: {apcer}\t BPCER: {bpcer}\t ACER: {acer}")
            writer.add_scalar('APCER', apcer, epoch)
            writer.add_scalar('BPCER', bpcer, epoch)
            writer.add_scalar('ACER', acer, epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
    writer.close()

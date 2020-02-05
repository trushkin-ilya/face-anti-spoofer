from torch.utils import data, tensorboard
from torch import nn
import torch
import numpy as np
from argparse import ArgumentParser
from datasets import CasiaSurfDataset
from torchvision import models
 
 
def evaluate(dataloader:data.DataLoader, model: nn.Module, loss_fn:nn.Module):
    model.eval()
    writer = tensorboard.SummaryWriter()
    print("Evaluating...")
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            labels = torch.LongTensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            acc = (torch.max(outputs.data, 1)[1] == labels).sum().item()
            val_acc.append(acc.item())
            val_loss.append(loss.item())
    avg_loss = np.mean(val_loss)
    avg_acc = np.mean(val_acc)
    print(
        f"\t\t\tValidation loss: {avg_loss}\t accuracy: {avg_acc}")
    writer.add_scalar('Validation loss', avg_loss, epoch)
    writer.add_scalar('Validation accuracy', avg_acc, epoch)
    writer.close()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--data-dir', type=str, default="./CASIA-SURF")
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--num_classes', type=int, default=2)
    args = argparser.parse_args()
    dataset = CasiaSurfDataset(args.protocol, dir=args.data_dir, train=False, transform=transforms.Resize((320, 240)))
    dataloader = data.DataLoader(dataset)
    model = models.mobilenet_v2(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint))
    loss_fn = nn.CrossEntropyLoss()
    evaluate(dataloader, model, loss_fn)
    

    
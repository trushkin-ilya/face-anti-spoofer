from torch.utils import data
from torch import nn
import torch
from argparse import ArgumentParser
from datasets import CasiaSurfDataset
from torchvision import models, transforms
import os
from sklearn import metrics


def evaluate(dataloader: data.DataLoader, model: nn.Module):
    model.eval()
    print("Evaluating...")
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            labels = torch.LongTensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(labels,
                                                                              torch.max(outputs.data, 1)[1]).ravel()
            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch
    apcer = fp / (tn + fp)
    bpcer = fn / (fn + tp)
    acer = (apcer + bpcer) / 2

    return apcer, bpcer, acer


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--data-dir', type=str, default=os.path.combine('data', 'CASIA_SURF'))
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1)
    args = argparser.parse_args()
    dataset = CasiaSurfDataset(args.protocol, dir=args.data_dir,
                               train=False, transform=transforms.Resize((320, 240)))
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size)
    model = models.mobilenet_v2(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    apcer, bpcer, acer = evaluate(dataloader, model)
    print(f'APCER: {apcer}, BPCER: {bpcer}, ACER: {acer}')

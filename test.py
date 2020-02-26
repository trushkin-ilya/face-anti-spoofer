import os
import torch

from argparse import ArgumentParser
from sklearn import metrics
from torch import nn
from torch.utils import data
from torchvision import models, transforms
from tqdm import tqdm
from datasets import CasiaSurfDataset


def evaluate(dataloader: data.DataLoader, model: nn.Module):
    device = next(model.parameters()).device
    model.eval()
    print("Evaluating...")
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            outputs = model(images.to(device))
            outputs = outputs.cpu()
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(y_true=labels,
                                                                              y_pred=torch.max(outputs.data, 1)[1],
                                                                              labels=[0, 1]).ravel()
            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch
    apcer = fp / (tn + fp) if fp != 0 else 0
    bpcer = fn / (fn + tp) if fn != 0 else 0
    acer = (apcer + bpcer) / 2

    return apcer, bpcer, acer


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--data-dir', type=str, default=os.path.join('data', 'CASIA_SURF'))
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0)
    args = argparser.parse_args()
    dataset = CasiaSurfDataset(args.protocol, mode='dev', dir=args.data_dir, transform=transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor()
    ]))
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = models.mobilenet_v2(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    apcer, bpcer, acer = evaluate(dataloader, model)
    print(f'APCER: {apcer}, BPCER: {bpcer}, ACER: {acer}')

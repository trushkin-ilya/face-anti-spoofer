from baseline.datasets import CasiaSurfDataset
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import models
import losses
import optimizers
import math
from tqdm import tqdm
import yaml

from transforms import TrainTransform


def find_lr(loader, net, optimizer, criterion, init_value=3e-10, final_value=3., beta=0.98):
    num = len(loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    for batch_num, data in enumerate(tqdm(loader)):
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))
        # Stop if the loss is exploding
        if batch_num > 0 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--config_path', required=True, type=str)
    argparser.add_argument('--optimizer', default="Adam", type=str)
    argparser.add_argument('--loss_fn', default="CrossEntropyLoss", type=str)
    argparser.add_argument('--num_classes', type=int, default=2)
    args = argparser.parse_args()
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    dataset = torch.utils.data.ConcatDataset(
        [CasiaSurfDataset(protocol, mode='train', depth=config['depth'], ir=config['ir'],
                          transform=TrainTransform()) for protocol in [1, 2, 3]])
    dataloader = data.DataLoader(
        dataset, sampler=data.sampler.RandomSampler(dataset), batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.figure(figsize=(21, 9))
    model = getattr(models, config['model'])(num_classes=args.num_classes)
    optimizer = getattr(optimizers, args.optimizer)(model.parameters())
    criterion = getattr(losses, args.loss_fn)()
    model = model.to(device)
    model.train()
    log_lrs, losses = find_lr(dataloader, model, optimizer, criterion)
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.show()

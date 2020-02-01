from datasets import CasiaSurfDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
from torchvision import models


def get_loader(dataset, batch_size=16, split=(4, 1)):
    val_q = len(dataset) // sum(split) * split[1]

    val_indices = np.random.choice(list(range(len(dataset))), val_q)
    train_indices = list(set(range(len(dataset))).difference(val_indices))
    sampler = SubsetRandomSampler(train_indices)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


if __name__ == '__main__':
    dataset = CasiaSurfDataset()
    data_loader = get_loader(dataset)
    model = models.mobilenet_v2(num_classes=2)
    model.classifier.add_module('2', torch.nn.Softmax(dim=0))
    optimizer = torch.optim.Adam(model.parameters())
    loss_fun = torch.nn.functional.binary_cross_entropy
    print(model)
    epochs = 10
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            images, labels = batch
            labels = np.array(labels, dtype=np.float)
            labels = torch.Tensor(labels)
            outputs = model(images)
            print(labels, outputs)
            loss = loss_fun(outputs, labels)
            print(f'Loss: {loss.item()} Epoch: {epoch + 1}')
            loss.backward()
            optimizer.step()

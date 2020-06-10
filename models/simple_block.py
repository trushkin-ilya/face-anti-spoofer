import torch

from torchvision.models.resnet import BasicBlock
from torch import nn


class SimpleBlock(nn.Module):
    def __init__(self, in_ch=3, norm_layer=nn.BatchNorm2d, num_classes=2, ch=64):
        super(SimpleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer = BasicBlock(ch, ch)
        self.fc = nn.Linear(ch * 56 * 56, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

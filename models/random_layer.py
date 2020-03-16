import torch

from torch import nn


class RandomLayer(nn.Module):
    def __init__(self, out_ch=2):
        super(RandomLayer, self).__init__()
        self.out_ch = out_ch

    def forward(self, x):
        return torch.rand((x.size(0), self.out_ch))

from baseline.models.MobileLiteNet import MobileLiteNet54
from torch import nn


def MobileLiteNet54_5ch(**kwargs):
    model = MobileLiteNet54(**kwargs)
    model.conv1 = nn.Conv2d(
        5, model.channels[0], kernel_size=3, stride=2, padding=1, bias=False)
    return model

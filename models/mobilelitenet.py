from baseline.models.MobileLiteNet import MobileLiteNet54
from torch import nn


def _MobileLiteNet54(in_ch: int, **kwargs):
    model = MobileLiteNet54(**kwargs)
    model.conv1 = nn.Conv2d(in_ch, model.channels[0], kernel_size=3, stride=2, padding=1, bias=False)
    return model

def MobileLiteNet54_4ch(**kwargs):
    return _MobileLiteNet54(4, **kwargs)

def MobileLiteNet54_5ch(**kwargs):
    return _MobileLiteNet54(5, **kwargs)

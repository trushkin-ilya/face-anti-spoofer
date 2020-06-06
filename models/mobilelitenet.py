from baseline.models.MobileLiteNet import MobileLiteNet54
from torch import nn


class MobileLiteNet54_5ch(MobileLiteNet54):
    def __init__(self, **kwargs):
        super(MobileLiteNet54_5ch, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(5, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False)

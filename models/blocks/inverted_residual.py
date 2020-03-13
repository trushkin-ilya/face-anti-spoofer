from torch import nn


# reference from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
class InvertedResidual(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride=1, expand_ratio=6, downsample=None):
        super(InvertedResidual, self).__init__()
        self.expansion = expand_ratio
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * self.expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if self.expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)

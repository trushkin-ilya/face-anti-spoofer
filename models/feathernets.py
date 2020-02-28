import torch
from torch import nn
import math

# reference form : https://github.com/moskomule/senet.pytorch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# reference from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
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

class MobileLiteNet(nn.Module):
    def __init__(self, block, layers, num_classes, se=False):

        super(MobileLiteNet, self).__init__()
        self.se = se
        self.channels = [32, 16, 32, 48, 64]
        self.inplanes = self.channels[1]
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels[1])

        self.layer1 = self._make_layer(block, self.channels[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, self.channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[4], layers[3], stride=2)
        self.final_DW = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=4,
                                  groups=self.channels[4], bias=False)
        self.do = nn.Dropout(0.2)
        self.linear = nn.Linear(self.channels[4] * 16, num_classes)
        if self.se:
            self.layer1_se1 = SELayer(self.channels[1])
            self.layer2_se2 = SELayer(self.channels[2])
            self.layer3_se3 = SELayer(self.channels[3])
            self.layer4_se4 = SELayer(self.channels[4])

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes,
                          kernel_size=3, stride=stride, padding=1, groups=self.inplanes, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.se:
            x = self.layer1_se1(self.layer1(x))
            x = self.layer2_se2(self.layer2(x))
            x = self.layer3_se3(self.layer3(x))
            x = self.layer4_se4(self.layer4(x))
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.final_DW(x)
        x = x.view(x.size(0), -1)

        x = self.do(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def MobileLiteNet54( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 6, 3], se=False, **kwargs)
    return model
def MobileLiteNet54_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 6, 3], se=True, **kwargs)
    return model
def MobileLiteNet102( **kwargs):
    model = MobileLiteNet(InvertedResidual, [3, 4, 23, 3], se=False, **kwargs)
    return model
def MobileLiteNet105_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 23, 3], se=True, **kwargs)
    return model
def MobileLiteNet153( **kwargs):
    model = MobileLiteNet(InvertedResidual, [3, 8, 36, 3], se=False, **kwargs)
    return model
def MobileLiteNet156_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 8, 36, 3], se=True, **kwargs)
    return model


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class FeatherNet(nn.Module):
    def __init__(self, num_classes=2, input_size=224, se=False, avgdown=False, width_mult=1.):
        super(FeatherNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2],  # 56x56
            [6, 48, 6, 2],  # 14x14
            [6, 64, 3, 2],  # 7x7
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                                                   nn.BatchNorm2d(input_channel),
                                                   nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
                                                   )
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample=downsample))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample=downsample))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        #         building last several layers
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                                groups=input_channel, bias=False)
                                      )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_DW(x)

        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def FeatherNetA(**kwargs):
    model = FeatherNet(se=True, **kwargs)
    return model


def FeatherNetB(**kwargs):
    model = FeatherNet(se=True, avgdown=True, **kwargs)
    return model


class Ensemble(nn.Module):
    def __init__(self, device, num_classes=2):
        super(Ensemble, self).__init__()

        self.num_classes = num_classes
        self.models = nn.ModuleList([FeatherNetA(num_classes=self.num_classes).to(device),
                                     FeatherNetB(num_classes=self.num_classes).to(device)])
        self.device = device

    def forward(self, x):
        output = torch.zeros([x.size(0), self.num_classes]).to(self.device)
        for model in self.models:
            output += model(x)
        return output

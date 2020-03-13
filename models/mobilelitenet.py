from torch import nn
from .blocks import SELayer
from .blocks import InvertedResidual
import math


class MobileLiteNet(nn.Module):
    def __init__(self, num_classes, block=InvertedResidual):
        layers = [4, 4, 6, 3]
        super(MobileLiteNet, self).__init__()
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


class MobileLiteNet_se(nn.Module):
    def __init__(self, num_classes, block=InvertedResidual):
        layers = [4, 4, 6, 3]
        super(MobileLiteNet_se, self).__init__()
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

        x = self.layer1_se1(self.layer1(x))
        x = self.layer2_se2(self.layer2(x))
        x = self.layer3_se3(self.layer3(x))
        x = self.layer4_se4(self.layer4(x))

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

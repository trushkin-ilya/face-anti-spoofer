from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class BasicDropoutBlock(BasicBlock):
    def __init__(self, *args, dropout=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_Dropout(ResNet):
    def __init__(self, **kwargs):
        super(ResNet18_Dropout, self).__init__(
            block=BasicDropoutBlock, layers=[2, 2, 2, 2], **kwargs)

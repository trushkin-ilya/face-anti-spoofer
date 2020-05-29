from torchvision.models import resnet18
from torch import nn

def resnet18_5ch(**kwargs):
    model = resnet18(**kwargs)
    model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
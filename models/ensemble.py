from torch import nn
from .mobile_lite_net import MobileLiteNet54_se, MobileLiteNet54
from .fishnet import fishnet150
from .feathernets import FeatherNetA, FeatherNetB
from torchvision import models


class Ensemble(nn.Module):
    def __init__(self, device, num_classes=2):
        super(Ensemble, self).__init__()

        self.num_classes = num_classes
        self.models = [FeatherNetA(num_classes=self.num_classes).to(device),
                       FeatherNetB(num_classes=self.num_classes).to(device),
                       fishnet150(num_cls=self.num_classes).to(device),
                       models.mobilenet_v2(num_classes=self.num_classes).to(device),
                       MobileLiteNet54(num_classes=self.num_classes).to(device),
                       MobileLiteNet54_se(num_classes=self.num_classes).to(device)]
        self.device = device

    def forward(self, x):
        return sum(map(lambda m: m(x), self.models))

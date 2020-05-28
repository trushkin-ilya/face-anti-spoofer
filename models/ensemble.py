from torch import nn
from baseline.models.MobileLiteNet import MobileLiteNet54, MobileLiteNet54_se
from .fishnet import FishNet150
from .feathernet import FeatherNet
from torchvision.models import MobileNetV2


class Ensemble(nn.Module):
    def __init__(self, device, num_classes=2):
        super(Ensemble, self).__init__()

        self.num_classes = num_classes
        self.models = [FeatherNet(se=True, num_classes=self.num_classes).to(device),
                       FeatherNet(se=True, avgdown=True,
                                  num_classes=self.num_classes).to(device),
                       FishNet150(num_cls=self.num_classes).to(device),
                       MobileNetV2(num_classes=self.num_classes).to(device),
                       MobileLiteNet54().to(device),
                       MobileLiteNet54_se().to(device)]
        self.device = device

    def forward(self, x):
        return sum(map(lambda m: m(x), self.models))

import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        loss = CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * loss

        return torch.mean(F_loss)

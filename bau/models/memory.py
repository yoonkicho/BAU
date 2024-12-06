from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_samples, momentum=0.1):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum

        self.register_buffer('features', torch.zeros(num_samples, num_features, dtype=torch.float))
        self.register_buffer('labels', torch.zeros(num_samples, num_features, dtype=torch.long))

    def momentum_update(self, inputs, targets):
        for x, y in zip(inputs, targets):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

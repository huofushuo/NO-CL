from __future__ import absolute_import

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.n_features = 0
        self._name = "BaseModule"

    def forward(self, x):
        return x

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def init_weights(self, std=0.01):
        print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(self), std))
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == '__main__':
    net = BaseModule()
    print(net)
    print("n_features:", net.n_features)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import extractor


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=True)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.shortcut = conv1x1(inplanes, planes, stride)
            self.use_conv1x1 = True
        else:
            self.use_conv1x1 = False

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)

        if self.use_conv1x1:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += shortcut

        return out


class WideResNet(extractor.BaseModule):
    def __init__(self, config, name):
        super(WideResNet, self).__init__()
        self.name = name
        depth = config["depth"]
        width = config["width"]
        dropout = config["dropout"]
        in_channels = config["channels"]
        print("%s depth: %d, width: %d, dropout=%f" \
              % (type(self), depth, width, dropout))

        layer = (depth - 4) // 6

        self.inplanes = 16
        self.conv = conv3x3(in_channels, 16)
        self.layer1 = self._make_layer(16*width, layer, dropout)
        self.layer2 = self._make_layer(32*width, layer, dropout, stride=2)
        self.layer3 = self._make_layer(64*width, layer, dropout, stride=2)
        self.bn = nn.BatchNorm2d(64*width)
        self.relu = nn.ReLU(inplace=True)

        self.n_features = 64 * width
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(64*width, num_classes)

        # use default init
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.weight.data = nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes, blocks, dropout, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(self.inplanes, planes, dropout, stride if i == 0 else 1))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x



if __name__ == '__main__':
    pass

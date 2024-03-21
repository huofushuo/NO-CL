from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import extractor


class LeNet(extractor.BaseModule):          
    def __init__(self, config, name):     
        super(LeNet, self).__init__()
        self.name = name
        in_channels = config["channels"]
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16, 
                               kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.n_features = 400
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x) 
        x = self.relu(self.conv2(x))
        return x

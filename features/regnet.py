import timm
from . import extractor

class RegNetX002(extractor.BaseModule):
    def __init__(self, config, name):
        super(RegNetX002, self).__init__()
       
        self.name = name
        self.features = timm.create_model('regnetx_002')
        self.n_features = 368

    def forward(self, x):
        return self.features.forward_features(x)

class RegNetY004(extractor.BaseModule):
    def __init__(self, config, name):
        super(RegNetY004, self).__init__()
       
        self.name = name
        self.features = timm.create_model('regnety_004')
        self.n_features = 440

    def forward(self, x):
        return self.features.forward_features(x)


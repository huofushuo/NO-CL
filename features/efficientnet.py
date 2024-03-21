import timm
from . import extractor

class EfficientNetB1(extractor.BaseModule):
    def __init__(self, config, name):
        super(EfficientNetB1, self).__init__()
       
        self.name = name
        self.features = timm.create_model('efficientnet_b1')
        self.n_features = 1280

    def forward(self, x):
        return self.features.forward_features(x)

class EfficientNetB0(extractor.BaseModule):
    def __init__(self, config, name):
        super(EfficientNetB0, self).__init__()
       
        self.name = name
        self.features = timm.create_model('efficientnet_b0')
        self.n_features = 1280

    def forward(self, x):
        return self.features.forward_features(x)

class EfficientNetV2S(extractor.BaseModule):
    def __init__(self, config, name):
        super(EfficientNetV2S, self).__init__()
       
        self.name = name
        drop_rate = config['dropout']
        self.features = timm.create_model('efficientnetv2_s', drop_rate=drop_rate) 
        self.n_features = 0  #FIXME: Rdit this

    def forward(self, x):
        return self.features(x)

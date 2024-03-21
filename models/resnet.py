"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import torch
from features.extractor import BaseModule
from scipy.linalg import hadamard
import math
from torch.autograd import Variable

class QNet(BaseModule):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )

    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt


class DVCNet(BaseModule):
    def __init__(self,
                 backbone,
                 n_units,
                 n_classes,
                 has_mi_qnet=True):
        super(DVCNet, self).__init__()

        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz,fea = self.backbone(xx)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]




def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet_DVC(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet_DVC, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.contiguous().view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits,out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.linear(out)
        return logits

class HadamardProj(nn.Module):
    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(sz))
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat.float())

        init_scale = 1. / math.sqrt(self.output_size)

        if fixed_scale is not None:
            self.scale = Variable(torch.Tensor(
                [fixed_scale]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.Tensor([init_scale]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                output_size).uniform_(-init_scale, init_scale))
        else:
            self.register_parameter('bias', None)

        self.eps = 1e-8

    def forward(self, x):
        if not isinstance(self.scale, nn.Parameter):
            self.scale = self.scale.type_as(x)
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.scale * \
            nn.functional.linear(x, w[:self.output_size, :self.input_size])
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out



class ResNet_PC(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, avg):
        super(ResNet_PC, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.hyper = projection_MLP(avg, 2048)
        self.hadamard_linear = HadamardProj(2048, 2048)
        self.linear = nn.Linear(2048, num_classes, bias=True)
        self.angular_linear = nn.Linear(2048, num_classes, bias=False)
        nn.init.xavier_uniform_(self.angular_linear.weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # ---- Tukey's transform
        beta = 0.5
        out = torch.pow(out[:, ], beta)
        return out

    def hyper_d(self, x):
        '''get hd logits'''
        x = self.hyper(x)
        return x

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        hd_out = self.hyper(out)
        logits = self.linear(hd_out)
        return logits

    def cos_forward(self, x):
        out = self.features(x)
        hd_out = self.hyper(out)
        cos_logits = F.linear(F.normalize(hd_out, p=2, dim=1), F.normalize(self.angular_linear.weight, p=2, dim=1))
        return cos_logits

def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def Reduced_ResNet18_PC(avg, nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet_PC(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, avg)

def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


def Reduced_ResNet18_DVC(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    backnone = ResNet_DVC(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)
    return DVCNet(backbone=backnone,n_units=128,n_classes=nclasses,has_mi_qnet=True)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # HadamardProj(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18(100)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)



class PCResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, avg=160 ,dim_in=2048, pre_class=80,  head='mlp', feat_dim=1024):
        super(PCResNet, self).__init__()
        self.encoder = Reduced_ResNet18_PC(avg, nclasses=pre_class)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def sc_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        feat = self.encoder.features(x)
        hd_feat = self.encoder.hyper_d(feat)
        if self.head:
            hd_feat = F.normalize(self.head(hd_feat), dim=1)
        else:
            hd_feat = F.normalize(hd_feat, dim=1)
        return hd_feat

    def gaa_features(self, x):
        feat = self.encoder.features(x)
        return feat

    def hd_features(self, feat):
        hd_feat = self.encoder.hyper_d(feat)
        return hd_feat

    def features(self, x):
        feat = self.encoder.features(x)
        hd_feat = self.encoder.hyper_d(feat)
        return hd_feat

    def linear_logits(self, x):
        '''true feature + logits'''
        logits = self.encoder.forward(x)
        return logits

    def cos_logits(self, x):
        '''true feature + cos_logits'''
        logits = self.encoder.cos_forward(x)
        return logits

    def pseudo_logits(self, x):  # x = feature pseudo
        '''pseudo logits'''
        logits = self.encoder.logits(x)
        return logits

    def hadamard_linear(self, x):
        hadamard_logits = self.encoder.hadamard_proj(x)
        return hadamard_logits

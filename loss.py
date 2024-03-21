import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy


def cross_entropy_loss(z, zt, ytrue, label_smoothing=0):
    zz = torch.cat((z, zt))
    yy = torch.cat((ytrue, ytrue))
    if label_smoothing > 0:
        ce = LabelSmoothingCrossEntropy(label_smoothing)(zz, yy)
    else:
        ce = nn.CrossEntropyLoss()(zz, yy)
    return ce


def cross_entropy(z, zt):
    # eps = np.finfo(float).eps
    Pz = F.softmax(z, dim=1)
    Pzt = F.softmax(zt, dim=1)
    # make sure no zero for log
    # Pz  [(Pz   < eps).data] = eps
    # Pzt [(Pzt  < eps).data] = eps
    return -(Pz * torch.log(Pzt)).mean()


def agmax_loss(y, ytrue, dl_weight=1.0):
    z, zt, zzt,_ = y
    Pz = F.softmax(z, dim=1)
    Pzt = F.softmax(zt, dim=1)
    Pzzt = F.softmax(zzt, dim=1)

    dl_loss = nn.L1Loss()
    yy = torch.cat((Pz, Pzt))
    zz = torch.cat((Pzzt, Pzzt))
    dl = dl_loss(zz, yy)
    dl *= dl_weight

    # -1/3*(H(z) + H(zt) + H(z, zt)), H(x) = -E[log(x)]
    entropy = entropy_loss(Pz, Pzt, Pzzt)
    return entropy, dl




def clamp_to_eps(Pz, Pzt, Pzzt):
    eps = np.finfo(float).eps
    # make sure no zero for log
    Pz[(Pz < eps).data] = eps
    Pzt[(Pzt < eps).data] = eps
    Pzzt[(Pzzt < eps).data] = eps

    return Pz, Pzt, Pzzt


def batch_probability(Pz, Pzt, Pzzt):
    Pz = Pz.sum(dim=0)
    Pzt = Pzt.sum(dim=0)
    Pzzt = Pzzt.sum(dim=0)

    Pz = Pz / Pz.sum()
    Pzt = Pzt / Pzt.sum()
    Pzzt = Pzzt / Pzzt.sum()

    # return Pz, Pzt, Pzzt
    return clamp_to_eps(Pz, Pzt, Pzzt)


def entropy_loss(Pz, Pzt, Pzzt):
    # negative entropy loss
    Pz, Pzt, Pzzt = batch_probability(Pz, Pzt, Pzzt)
    entropy = (Pz * torch.log(Pz)).sum()
    entropy += (Pzt * torch.log(Pzt)).sum()
    entropy += (Pzzt * torch.log(Pzzt)).sum()
    entropy /= 3
    return entropy



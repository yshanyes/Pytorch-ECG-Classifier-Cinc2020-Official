# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''

from .mixnet_sm import mixnet_sm_pretrain, mixnet_sm, mixnet_sm_predict
from .mixnet_mm import mixnet_mm_pretrain, mixnet_mm, mixnet_mm_predict
from .resnet import resnet34
from .senet import seresnet34, seresnet50,seresnet101,seresnext26_32x4d,seresnext50_32x4d
from .resnest import resnest50,resnest101
from .iresnest import iresnest50_predict,iresnest101,iresnest50_pretrain
from .semknet import semkresnet34,semkresnet18
from .multi_scale_resnet import MSResNet

# from .utils import SelectAdaptivePool1d

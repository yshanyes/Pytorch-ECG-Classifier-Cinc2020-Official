import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Module, Linear, BatchNorm1d, ReLU
from torch.nn.modules.utils import _pair
# ref:https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnest.py
# ref:https://github.com/iduta/iresnet/blob/master/models/iresnet.py
from config import config
__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269', 'ResNet', 'Bottleneck', 'SKConv1d']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

class GroupNorm1d(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm1d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H)
        return x * self.weight + self.bias

# ref:https://github.com/c0nn3r/pytorch_highway_networks/blob/master/main.py
class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)

class DropBlock1D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SplAtConv1d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv1d, self).__init__()

        padding = padding#1#_pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv1d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv1d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv1d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv1d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock1D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool1d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


class DropBlock1D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool1d(inputs, 1).view(inputs.size(0), -1)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        """Global max pooling over the input's spatial dimensions"""
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_max_pool1d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(Bottleneck, self).__init__()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv1d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool1d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock1D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock1D(dropblock_prob, 3)
            self.dropblock3 = DropBlock1D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv1d(
                group_width, group_width, kernel_size=5,
                stride=stride, padding=2,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv1d(
                group_width, group_width, kernel_size=5, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv1d(
                group_width, group_width, kernel_size=5, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv1d(
            group_width, planes * 4, kernel_size=1, bias=False)

        #self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)

        if start_block:
            self.bn3 = norm_layer(planes * self.expansion)

        if end_block:
            self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            if self.dropblock_prob > 0.0:
                out = self.dropblock3(out)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            # if self.dropblock_prob > 0.0:
            #     out = self.dropblock3(out)
            out = self.relu(out)

###############################################

        # residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # if self.dropblock_prob > 0.0:
        #     out = self.dropblock1(out)
        # out = self.relu(out)

        # if self.avd and self.avd_first:
        #     out = self.avd_layer(out)
        # out = self.conv2(out)
        # if self.radix == 1:
        #     out = self.bn2(out)
        #     if self.dropblock_prob > 0.0:
        #         out = self.dropblock2(out)
        #     out = self.relu(out)

        # if self.avd and not self.avd_first:
        #     out = self.avd_layer(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # if self.dropblock_prob > 0.0:
        #     out = self.dropblock3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=27, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,max_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.5, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm1d):#
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.max_down = max_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv1d
        else:
            conv_layer = nn.Conv1d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                #conv_layer(12, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                conv_layer(12, stem_width, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs),#0.856
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
#         self.avgpool = GlobalAvgPool1d()
        self.glbmaxpool = GlobalMaxPool1d()

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion , num_classes)

#         self.highway_number = 6

#         self.highway_layers = nn.ModuleList([HighwayMLP(512 * block.expansion,
#                                                         activation_function=F.relu,
#                                                         gate_activation=F.sigmoid)
#                                              for _ in range(self.highway_number)])
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        #############################################################
        #                            IResNet                        #
        #############################################################
        if self.max_down:
            if stride != 1 and self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                    #conv1x1(self.inplanes, planes * block.expansion),
                    nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes * block.expansion),
                )
            elif self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    #conv1x1(self.inplanes, planes * block.expansion),
                    nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes * block.expansion),
                )
            elif stride != 1:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                    norm_layer(planes * block.expansion),
                )

        #############################################################
        #                            ResNeSt                        #
        #############################################################
        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                down_layers = []
                if self.avg_down:
                    if dilation == 1:
                        down_layers.append(nn.AvgPool1d(kernel_size=stride, stride=stride,
                                                        ceil_mode=True, count_include_pad=False))
                    else:
                        down_layers.append(nn.AvgPool1d(kernel_size=1, stride=1,
                                                        ceil_mode=True, count_include_pad=False))
                    down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=1, bias=False))
                else:
                    down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False))

                down_layers.append(norm_layer(planes * block.expansion))
                downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,start_block=True))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,start_block=True))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion

        exclude_bn0 = True
        for i in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(block(self.inplanes, planes,
                        radix=self.radix, cardinality=self.cardinality,
                        bottleneck_width=self.bottleneck_width,
                        avd=self.avd, avd_first=self.avd_first,
                        dilation=dilation, rectified_conv=self.rectified_conv,
                        rectify_avg=self.rectify_avg,
                        norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                        last_gamma=self.last_gamma,end_block=True,exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = self.glbmaxpool(x)

        #x = torch.cat([self.avgpool(x), self.glbmaxpool(x)], dim=1)
        #x = x.view(x.size(0), -1)

        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)

#         for current_layer in self.highway_layers:
#             x = current_layer(x)

        x = self.fc(x)

        return x

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def iresnest50_predict(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=27,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, max_down=True,#avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    if pretrained:
        print("iresnest50 pretrain")#"./round2/mixnet_mm_not_all_data_transform_best_weight.pth"
        model.load_state_dict(torch.load(config.round1_pretrain_weight, map_location='cpu')['state_dict'])
    return model
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))
    # return model

def iresnest50_pretrain(pretrained=True, root='~/.encoding/models', fold=None,**kwargs):
    if fold != None:
        model = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=34,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, max_down=True,#avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=55,
               radix=2, groups=1, bottleneck_width=64,
               deep_stem=True, stem_width=32, max_down=True,#avg_down=True,
               avd=True, avd_first=False, **kwargs)

    if pretrained:
        if fold != None:
            print("iresnest50 pretrain cv")#"./round2/mixnet_mm_not_all_data_transform_best_weight.pth"
            model.load_state_dict(torch.load(config.round1_pretrain_weight_cv.format(fold), map_location='cpu')['state_dict'])
        else:
            print("iresnest50 pretrain")#"./round2/mixnet_mm_not_all_data_transform_best_weight.pth"
            model.load_state_dict(torch.load(config.round1_pretrain_weight, map_location='cpu')['state_dict'])
    return model
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))
    # return model

def iresnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def iresnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def iresnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model

if __name__ == '__main__':

    # x = torch.randn(2,12,256*50)
    x = torch.randn(2,12,500*10)
    m = iresnest50()
    print(m)
    print(m(x))

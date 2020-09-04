import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config

import math
#https://github.com/romulus0914/MixNet-Pytorch/blob/master/mixnet.py

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm1d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv1d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm1d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv1d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv1d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm1d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv1d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm1d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv1d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm1d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=34):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=3, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True)
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        # input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj
        # self.a3 = Inception(192, 64,  96,  128, 16, 32, 32)
        # self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.a3 = Inception(12,  24,  12,  24, 24, 24, 24)
        self.b3 = Inception(96, 32, 32, 32, 24, 24, 24)
        #"""In general, an Inception network is a network consisting of
        #modules of the above type stacked upon each other, with occasional 
        #max-pooling layers with stride 2 to halve the resolution of the 
        #grid"""
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool1d(2, stride=1, padding=1)

        self.a4 = Inception(112, 32, 64,  32, 12, 24, 32)
        self.b4 = Inception(120, 32, 64, 64, 24, 32, 32)
        # self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        # self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        # self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        # self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        # self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.lstm = nn.LSTM(300, 112, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(224, 64, bidirectional=True, batch_first=True)   

        self.attention_layer = Attention(128, 112)

        self.linear = nn.Linear(128, num_class)
    
    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)

        output = self.b3(output)
        
        # output = self.maxpool(output)
        # output = self.a4(output)

        # output = self.b4(output)
        # output = self.c4(output)
        # output = self.d4(output)
        # output = self.e4(output)

        # output = self.maxpool(output)

        # output = self.a5(output)
        # output = self.b5(output)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%, 
        #however the use of dropout remained essential even after 
        #removing the fully connected layers."""
        # output = self.avgpool(output)
        # output = self.dropout(output)
        # max_pooled = F.adaptive_max_pool1d(output, 1)
        # avg_pooled = F.adaptive_avg_pool1d(output, 1)
        # output1 = torch.cat([max_pooled, avg_pooled], dim=1)

        # output1 = output1.view(output1.size(0), -1)


        output,_ = self.lstm(output)
        output,_ = self.gru(output)

        output = self.attention_layer(output)
        output = output.view(output.size(0), -1)
        # output = torch.cat([output, output1], dim=1)

        # output = output.view(output.size()[0], -1)
        # output = self.linear(output)

        return output

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm1d(out_channels),
        NON_LINEARITY[non_linear]
    )

def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm1d(out_channels),
        NON_LINEARITY[non_linear]
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv1d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv1d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, 2, keepdim=True)#(2, 3),
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class GroupedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv1d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv1d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv1d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MixNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3],
        expand_ksize=[1],
        project_ksize=[1],
        stride=1,
        expand_ratio=1,
        non_linear='ReLU',
        se_ratio=0.0
    ):

        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv1d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm1d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm1d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv1d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm1d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MixNet(nn.Module):
    # [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
    mixnet_s = [(16,  16,  [3],              [1],    [1],    1, 1, 'ReLU',  0.0),
                (16,  24,  [3],              [1, 1], [1, 1], 2, 6, 'ReLU',  0.0),
                (24,  24,  [3],              [1, 1], [1, 1], 1, 3, 'ReLU',  0.0),
                (24,  40,  [3, 5, 7],        [1],    [1],    2, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],           [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],           [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],           [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  80,  [3, 5, 7],        [1],    [1, 1], 2, 6, 'Swish', 0.25),
                (80,  80,  [3, 5],           [1],    [1, 1], 1, 6, 'Swish', 0.25),
                (80,  80,  [3, 5],           [1],    [1, 1], 1, 6, 'Swish', 0.25),
                (80,  120, [3, 5, 7],        [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9],     [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9],     [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9, 11], [1],    [1],    2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9],     [1],    [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9],     [1],    [1, 1], 1, 6, 'Swish', 0.5)]
    
    mixnet_m = [(24,  24,  [3],          [1],    [1],    1, 1, 'ReLU',  0.0),
                (24,  32,  [3, 5, 7],    [1, 1], [1, 1], 2, 6, 'ReLU',  0.0),
                (32,  32,  [3],          [1, 1], [1, 1], 1, 3, 'ReLU',  0.0),
                (32,  40,  [3, 5, 7, 9], [1],    [1],    2, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40,  80,  [3, 5, 7],    [1],    [1],    2, 6, 'Swish', 0.25),
                (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80,  120, [3],          [1],    [1],    1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], [1],    [1],    2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, 'Swish', 0.5)]

    def __init__(self, net_type='mixnet_s', beat_net_type='mixnet_m',input_size=2560, num_classes=34, stem_channels=16,
                 feature_size=256, depth_multiplier=1.0,beat_feature_size=256, beat_depth_multiplier=1.0):
        super(MixNet, self).__init__()

        if net_type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
            dropout_rate = 0.2
        elif net_type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
            dropout_rate = 0.25
        elif net_type == 'mixnet_l':
            config = self.mixnet_m
            stem_channels = 24
            depth_multiplier *= 1.3
            dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')
        # for beat 
        if beat_net_type == 'mixnet_s':
            beat_config = self.mixnet_s
            beat_stem_channels = 16
            beat_dropout_rate = 0.2
        elif beat_net_type == 'mixnet_m':
            beat_config = self.mixnet_m
            beat_stem_channels = 24
            beat_dropout_rate = 0.25
        elif beat_net_type == 'mixnet_l':
            beat_config = self.mixnet_m
            beat_stem_channels = 24
            beat_depth_multiplier *= 1.3
            beat_dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')

        assert input_size % 32 == 0

        # depth multiplier
        if depth_multiplier != 1.0:
            stem_channels = _RoundChannels(stem_channels*depth_multiplier)

            for i, conf in enumerate(config):
                conf_ls = list(conf)
                conf_ls[0] = _RoundChannels(conf_ls[0]*depth_multiplier)
                conf_ls[1] = _RoundChannels(conf_ls[1]*depth_multiplier)
                config[i] = tuple(conf_ls)

        # for beat 
        # depth multiplier
        if beat_depth_multiplier != 1.0:
            beat_stem_channels = _RoundChannels(beat_stem_channels*beat_depth_multiplier)

            for i, conf in enumerate(beat_config):
                beat_conf_ls = list(conf)
                beat_conf_ls[0] = _RoundChannels(beat_conf_ls[0]*beat_depth_multiplier)
                beat_conf_ls[1] = _RoundChannels(beat_conf_ls[1]*beat_depth_multiplier)
                beat_config[i] = tuple(beat_conf_ls)
        # stem convolution
        self.stem_conv = Conv3x3Bn(12, stem_channels, 2)

        # for beat 
        # stem convolution
        self.beat_stem_conv = Conv3x3Bn(12, beat_stem_channels, 2)

        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                expand_ksize=expand_ksize,
                project_ksize=project_ksize,
                stride=stride,
                expand_ratio=expand_ratio,
                non_linear=non_linear,
                se_ratio=se_ratio
            ))
        self.layers = nn.Sequential(*layers)

        # for beat 
        # building MixNet blocks
        beat_layers = []
        for beat_in_channels, beat_out_channels, beat_kernel_size, beat_expand_ksize, beat_project_ksize, beat_stride, \
                beat_expand_ratio, beat_non_linear, beat_se_ratio in beat_config:
            beat_layers.append(MixNetBlock(
                beat_in_channels,
                beat_out_channels,
                kernel_size=beat_kernel_size,
                expand_ksize=beat_expand_ksize,
                project_ksize=beat_project_ksize,
                stride=beat_stride,
                expand_ratio=beat_expand_ratio,
                non_linear=beat_non_linear,
                se_ratio=beat_se_ratio
            ))
        self.beat_layers = nn.Sequential(*beat_layers)

        # last several layers
        self.head_conv = Conv1x1Bn(config[-1][1], feature_size)

        # for beat 
        # last several layers
        self.beat_head_conv = Conv1x1Bn(beat_config[-1][1], beat_feature_size)

        self.googlenet = GoogleNet()

        #self.avgpool = nn.AvgPool1d(input_size//32, stride=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(1024, num_classes)#
        # self.dropout = nn.Dropout(dropout_rate)

        self._initialize_weights()

    def forward(self, x, beat):#
        # print(beat.shape)
        # beat = x[:,:,-256:]
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)

        
        # print(beat.shape)
        # beat = self.googlenet(beat)
        beat = self.beat_stem_conv(beat)
        beat = self.beat_layers(beat)
        beat = self.beat_head_conv(beat)
        # print(x.shape)
        # print(beat.shape)

        # print(x.shape)
        # x = self.avgpool(x)
        max_pooled = F.adaptive_max_pool1d(x, 1)
        avg_pooled = F.adaptive_avg_pool1d(x, 1)
        x = torch.cat([max_pooled, avg_pooled], dim=1)
        x = x.view(x.size(0), -1)


        beat = torch.cat([F.adaptive_max_pool1d(beat, 1), F.adaptive_avg_pool1d(beat, 1)], dim=1)
        beat = beat.view(beat.size(0), -1)

        # print(x.shape)
        # print(beat.shape)
        x = torch.cat([x, beat], dim=1)
        x = self.classifier(x)
        # x = self.dropout(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# def mixnet_s(pretrained=False, **kwargs):

#     model = MixNet(net_type='mixnet_s')
#     return model

# def mixnet_m(pretrained=False, **kwargs):

#     model = MixNet(net_type='mixnet_m')
#     return model

# def mixnet_l(pretrained=False, **kwargs):

#     model = MixNet(net_type='mixnet_l')
#     return model

def mixnet_sm_pretrain(pretrained=False, **kwargs):#True

    model = MixNet(net_type='mixnet_s',beat_net_type='mixnet_m',num_classes=55)
    if pretrained:
        print("pretrain")#    #mixnet_sm_all_data_transform_best_weight
        model.load_state_dict(torch.load("./round2/mixnet_sm_transform_best_weight.pth", map_location='cpu')['state_dict'])
    return model

def mixnet_sm(pretrained=True, **kwargs):#True

    model = MixNet(net_type='mixnet_s',beat_net_type='mixnet_m',num_classes=55)
    if pretrained:
        print("pretrain")#    #mixnet_sm_all_data_transform_best_weight#"./round2/mixnet_sm_transform_best_weight.pth"
        model.load_state_dict(torch.load(config.round1_pretrain_mixnet_sm_weight, map_location='cpu')['state_dict'])
    return model


def mixnet_sm_predict(pretrained=False, **kwargs):#True

    model = MixNet(net_type='mixnet_s',beat_net_type='mixnet_m',num_classes=34)
    if pretrained:
        print("pretrain")#    #mixnet_sm_all_data_transform_best_weight
        model.load_state_dict(torch.load("./round2/mixnet_sm_transform_best_weight.pth", map_location='cpu')['state_dict'])
    return model

if __name__ == '__main__':
    net = mixnet_sm()
    print(net)
    x = Variable(torch.randn(10, 12, 2560))
    x_beat = Variable(torch.randn(10, 12, 300))
    y = net(x,x_beat)
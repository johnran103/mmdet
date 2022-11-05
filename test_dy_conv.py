
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32


import nltk
from nltk.cluster.kmeans import KMeansClusterer

from mmdet.core import (anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.core.utils import filter_scores_and_topk


class attention1d(nn.Module):
    def __init__(self, in_planes=1, ratios=16, K=4, temperature=1, init_weight=True): # quality map
        super(attention1d, self).__init__()
        assert temperature % 3 == 1
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        self.K = K
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        _N, _C, _H, _W = x.size()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    '''
    Args:
        x(Tensor):  shape (batch, in_channel, height, width)
        quality_map(Tensor):  shape (batch, 1, height, width)
    
    Return:
        output(Tensor):  shape (batch, out_channel, height, width)
    

    Note:
        in_channel must eqal to out_channel
    '''


    def __init__(self, in_planes, out_planes, ratio=16.0, stride=1, padding=0, dilation=1, bias=True, K=2,temperature=1, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.K = K
        self.attention = attention1d(1, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self): # maybe problematic
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, quality_map):# a different version of dynamic convlution, is another kind of spatial attention
        residule = x
        batch_size, in_planes, height, width = x.size()

        softmax_attention = self.attention(quality_map)
        print(f'attention size {softmax_attention.size()}')
        print(f'attention {softmax_attention}')

        softmax_attention = softmax_attention.permute(0, 2, 3, 1)

        print(f'attention size after {softmax_attention.size()}')
        print(f'attention after {softmax_attention}')

        #x = x.view(1, -1, width, height)# 变化成一个维度进行组卷积
        #weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同)
        #weight = weight.view(self.K, self.in_planes, self.out_planes)
        # print(f'softmax_attention {softmax_attention.size()}')
        # print(f'self.weight {self.weight.size()}')
        weight = self.weight.view(self.K, -1)

        print(f'weight size {weight.size()}')
        print(f'weight {weight}')

        aggregate_weight = torch.matmul(softmax_attention, weight).view(batch_size, height, width, self.out_planes, self.in_planes)# (N, H, W, C2, C1)
        
        print(f'aggregate_weight size {aggregate_weight.size()}')
        print(f'aggregate_weight {aggregate_weight}')
        
        aggregate_weight = aggregate_weight.permute(3, 0, 4, 1, 2)  # (C2, N, C1, H, W)

        print(f'aggregate_weight after size {aggregate_weight.size()}')
        print(f'aggregate_weight after {aggregate_weight}')

        output = aggregate_weight * x[None, :, :, :, :]
        
        # if self.bias is not None:
        #     aggregate_bias = torch.matmul(softmax_attention, self.bias).permute(0, 3, 1, 2) # (N, C1, H, W)
        #     print(aggregate_bias.size())
        #     print(softmax_attention.size())
        #     output = output + aggregate_bias


        output = output.sum(dim=0) # (N, C1, H, W)
        return residule + output



dy1 = Dynamic_conv1d(2, 1)

x = torch.tensor([[[[1, 2],[3, 4]],[[5, 6],[7, 8]]]], dtype=torch.float32)
y = torch.tensor([[[[1,2],[3,4]]]], dtype=torch.float32)

print(f'x size {x.size()}')
print(f'x {x}')
print(f'y size {y.size()}')
print(f'y {y}')

result = dy1(x, y)
print(f'output size {result.size()}')
print(f'output {result}')
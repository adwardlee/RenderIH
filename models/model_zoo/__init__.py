import torch.nn as nn

from .fc import build_fc_layer
from .hrnet import get_hrnet, Bottleneck
from .coarsening import build_graph
from .graph_utils import graph_upsample, graph_avg_pool

__all__ = ['build_fc_layer', 'get_hrnet', 'Bottleneck',
           'build_graph', 'GCN_vert_convert', 'graph_upsample', 'graph_avg_pool',
           'weights_init', 'conv1x1', 'conv3x3', 'deconv3x3']


class noop(nn.Module):
    def forward(self, x):
        return x


def build_activate_layer(actType):
    if actType == 'relu':
        return nn.ReLU(inplace=True)
    elif actType == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif actType == 'elu':
        return nn.ELU(inplace=True)
    elif actType == 'sigmoid':
        return nn.Sigmoid()
    elif actType == 'tanh':
        return nn.Tanh()
    elif actType == 'noop':
        return noop()
    else:
        raise RuntimeError('no such activate layer!')


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class unFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)


def conv1x1(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
              build_activate_layer(actFun),
              bn]
    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
              build_activate_layer(actFun),
              bn]
    return nn.Sequential(*layers)


def deconv3x3(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        build_activate_layer(actFun),
        bn
    )


class GCN_vert_convert():
    def __init__(self, vertex_num=1, graph_perm_reverse=[0], graph_perm=[0]):
        self.graph_perm_reverse = graph_perm_reverse[:vertex_num]
        self.graph_perm = graph_perm

    def vert_to_GCN(self, x):
        # x: B x v x f
        return x[:, self.graph_perm]

    def GCN_to_vert(self, x):
        # x: B x v x f
        return x[:, self.graph_perm_reverse]

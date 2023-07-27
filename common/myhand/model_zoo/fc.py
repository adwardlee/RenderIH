import torch.nn as nn


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


def build_fc_layer(inDim, outDim, actFun='relu', dropout_prob=-1, weight_norm=False):
    net = []
    if dropout_prob > 0:
        net.append(nn.Dropout(p=dropout_prob))
    if weight_norm:
        net.append(nn.utils.weight_norm(nn.Linear(inDim, outDim)))
    else:
        net.append(nn.Linear(inDim, outDim))
    net.append(build_activate_layer(actFun))
    return nn.Sequential(*net)
